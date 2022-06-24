import wandb

import math
import os
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

import wandb

from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
from env import Mujoco

def train(rank, cfg, shared_model, exp_time):

    # seed 고정
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # env
    mujoco = Mujoco(cfg.env_name)

    # model
    model = ActorCritic(mujoco.env.observation_space.shape[0], mujoco.env.action_space.shape[0])
    model.train()

    # optimizer
    optimizer = optim.Adam(shared_model.parameters(), lr = cfg.lr)

    if rank == 0:
        wandb.init(project='a3c', entity = "minjuling",name=cfg.env_name+'train'+exp_time)
        print(mujoco.env.observation_space.shape, mujoco.env.action_space.shape)

        # wandb.watch(shared_model, log = "gradients", log_freg = 1000)
        # wandb.config.update(str(cfg))


    state = mujoco.env.reset()
    state = torch.from_numpy(state)
    done = True

    step_len = 0
    epi_len = 0

    while True:
        step_len += 1
        # 새로운 에피소드 마다 shared model 불러옴
        model.load_state_dict(shared_model.state_dict())
        if done: 
            cx = torch.zeros(1,128)
            hx = torch.zeros(1,128)
        else:
            cx = cx.data
            hx = hx.data

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(cfg.num_steps):
            value, mu, sigma, (hx, cx) = model(
                (state.unsqueeze(0).float(), (hx, cx)))

            # action select
            action_dist = dist.normal.Normal(mu, sigma)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy() 
            # uniform dist를 뽑지 않고 sigma가 더 작아지게 도와줌           

            # env step
            state, reward, done, _ = mujoco.env.step(action.numpy())
            done = done or step_len >= cfg.max_step_len

            state = torch.from_numpy(state) # s'
            values.append(value) # s 에서의 v값
            log_probs.append(log_prob) # log pi(a|s)
            rewards.append(reward)
            entropies.append(entropy) # entropy

            if done:
                epi_len += 1
                step_len = 0
                state = mujoco.env.reset()
                state = torch.from_numpy(state)
                break
        
        R = 0
        if not done:
            value, _, _, _ = model((state.unsqueeze(0).float(), (hx, cx)))
            R = value.data
        
        values.append(R)
        policy_loss = 0
        value_loss = 0

        # calculate the rewards from the terminal state
        for i in reversed(range(len(rewards))):
            R = cfg.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + advantage.pow(2)

            policy_loss = policy_loss - (log_probs[i]*advantage.detach()).sum() \
                        - (0.0001*entropies[i]).sum()

        if rank == 0 and  epi_len%cfg.loss_frequency == 0:
            wandb.log({'policy_loss' : policy_loss})
            wandb.log({'value_loss' : value_loss})
        
        optimizer.zero_grad()

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            shared_param._grad = param.grad
        optimizer.step()




