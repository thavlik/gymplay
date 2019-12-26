import gym
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import torchcontrib
from torchcontrib.optim import SWA
import time
import requests
import numpy as np
from collections import namedtuple


import os
import torchvision
import math
import argparse

device = torch.device('cpu')
torch.autograd.set_detect_anomaly(True)

class MyModel(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 layers=[64, 32, 16],
                 dropout=0.1):
        super(MyModel, self).__init__()
        l = []
        features = input_features
        for layer in layers:
            #l.append(nn.BatchNorm1d(features))
            l.append(nn.Linear(features, layer))
            l.append(nn.ReLU())
            features = layer
        #l.append(nn.BatchNorm1d(features))
        l.append(nn.Dropout(dropout))
        l.append(nn.Linear(features, output_features))
        l.append(nn.ReLU())
        l.append(nn.Softmax(dim=-1))
        self.seq = nn.Sequential(*l)

        self.value1 = nn.Linear(input_features, 128)
        self.value2 = nn.Linear(128, 1)

    def forward(self, input):
        action = input
        action = self.seq(action)

        # value prediction - critic
        value = F.selu(self.value1(input))
        value = self.value2(value)

        return action, value

ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])

class AAC(MyModel):
    def __init__(self, args):
        envs = [gym.make('LunarLander-v2') for _ in range(args.batch_size)]
        obs_features = envs[0].observation_space.shape[0]
        num_actions = envs[0].action_space.n
        super(AAC, self).__init__(input_features=obs_features,
                                  output_features=num_actions)
        self.training = args.mode == 'train'
        self.envs = envs
        self.outputs = []
        self.rewards = []
        self.discount = args.episode_discount
        self.discounts = []
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        optimizer = SWA(optimizer, swa_start=64, swa_freq=128, swa_lr=0.05)
        optimizer.defaults = {}
        if os.path.isfile(args.checkpoint):
            model_dict = torch.load(args.checkpoint)
            self.load_state_dict(model_dict)
            print(f'Loaded {args.checkpoint}')
        if os.path.isfile(args.checkpoint + '_optimizer'):
            optimizer_dict = torch.load(args.checkpoint + '_optimizer')
            optimizer.load_state_dict(optimizer_dict)
            print(f'Loaded {args.checkpoint}_optimizer')
        self.optimizer = optimizer

    def reset(self):
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def get_action(self, obs):
        action = self.forward(obs)
        return action

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)

    def set_terminal(self, terminal):
        self.discounts.append(self.discount * terminal)

    def forward(self, input):
        action_prob, value = super(AAC, self).forward(input)
        _, action = action_prob.max(1, keepdim=True)
        if not self.training:
            return action
        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action

    def backward(self):
        #
        # calculate step returns in reverse order
        rewards = self.rewards

        returns = torch.Tensor(len(rewards) - 1, *self.outputs[-1].value.size())
        step_return = self.outputs[-1].value.detach().cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discounts[i]).add_(rewards[i])
            returns[i] = step_return

        returns = returns.to(device)
        #
        # calculate losses
        policy_loss = 0
        value_loss = 0
        steps = len(self.outputs) - 1
        for i in range(steps):
            advantage = returns[i] - self.outputs[i].value.detach()
            policy_loss += -self.outputs[i].log_action * advantage
            value_loss += F.smooth_l1_loss(self.outputs[i].value, returns[i])

        weights_l2 = 0
        for param in self.parameters():
            weights_l2 += param.norm(2)

        loss = policy_loss.mean()/steps + value_loss/steps + 0.00001*weights_l2
        loss.backward()

        # reset state
        self.reset()

    def run_test(self, args):
        env = self.envs[0]
        self.optimizer.swap_swa_sgd() 
        episode = 0
        while True:
            obs = env.reset()
            for step in range(args.steps_per_episode):
                obs = torch.Tensor(obs).view(1, obs.shape[0]).to(device)
                action = self.get_action(obs).to(torch.device('cpu'))
                obs, reward, term, _ = env.step(action.numpy()[0][0])
                env.render()
                print(f'{step} action: {action} reward: {reward}')
                if term:
                    break
            episode += 1

    def run_train(self, args):
        envs, optimizer = self.envs, self.optimizer
        episode = 0
        while True:
            optimizer.zero_grad()
            batch_time = time.time()
            obs = [env.reset() for env in envs]
            reward = [None for _ in envs]
            terminal = [0 for _ in envs]
            episode_return = [0 for _ in envs]
            for _ in range(args.steps_per_episode):
                action = self.get_action(torch.Tensor(obs).to(device)).to(torch.device('cpu'))
                def step(i, env):
                    obs[i], _reward, term, _ = env.step(action[i].numpy()[0])
                    reward[i] = _reward
                    episode_return[i] += _reward
                    terminal[i] = 1 if not term else 0
                    #print(f'{i} action: {action[i]} reward: {_reward} term: {term}')
                [step(i, env) for i, env in enumerate(envs)]
                self.set_reward(torch.Tensor(reward).view(args.batch_size, 1))
                self.set_terminal(torch.Tensor(terminal).view(args.batch_size, 1))
                if args.render:
                    envs[0].render()
            self.backward()

            grads = []
            weights = []
            for p in self.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                    weights.append(p.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            assert grads_norm == grads_norm

            optimizer.step()
            optimizer.zero_grad()

            mean_return = torch.Tensor(episode_return).mean()

            if episode % 1 == 0:
                print("{}: return: {:f}, mean_return = {:f}, grads_norm = {:f}, weights_norm = {:f}, batch_time = {:.3f}".format(episode, episode_return[0], mean_return, grads_norm, weights_norm, time.time()-batch_time))
            if episode % args.save_epoch == 0:
                print(f'Saving checkpoint...')
                torch.save(self.state_dict(), args.checkpoint)
                torch.save(optimizer.state_dict(), args.checkpoint+'_optimizer')
            episode += 1

        [env.close() for env in envs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI Gym Trainer')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='number of game instances running in parallel')
    parser.add_argument('--seed', type=int, default=1, help='seed value')
    parser.add_argument('--save_epoch', type=int, default=128, help='num epochs per checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='checkpoint file path')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--render', type=bool, default=False, help='render training to screen')
    parser.add_argument('--steps_per_episode',
                        type=int,
                        default=100,
                        help='steps per episode')
    parser.add_argument('--episode_discount',
                        type=float,
                        default=0.95,
                        help='')
    args = parser.parse_args()
    print(args)

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = AAC(args).to(device)
    if args.mode == 'train':
        model.run_train(args)
    elif args.mode == 'test':
        model.run_test(args)
    
