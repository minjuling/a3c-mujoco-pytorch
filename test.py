import wandb

import math
import os
import numpy as np
import random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
from env import Mujoco

import wandb

def get_log_name():
    now = time.localtime()
    return str(now.tm_mon) +'_' + str(now.tm_mday)+ '_' + str(now.tm_hour) + '_'+ str(now.tm_min)

def test(rank, cfg, shared_model, exp_time):
    wandb.init(project='a3c', entity = "minjuling",name=cfg.env_name+'test'+exp_time)
    # wandb.config.update(cfg)

    # seed 고정
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # env
    mujoco = Mujoco(cfg.env_name)
    mujoco.env.seed(cfg.seed)

    # model
    model = ActorCritic(mujoco.env.observation_space.shape[0], mujoco.env.action_space.shape[0])
    model.eval()


    # optimizer
    optimizer = optim.Adam(shared_model.parameters(), lr = cfg.lr)

    state = mujoco.env.reset()
    state = torch.from_numpy(state)
    total_reward = 0
    done = True
    step_len = 0
    epi_len = 0
    reward_list = []
    best_reward = 100

    start_time = time.time()
    
    while True:
        step_len += 1
        if done: 
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1,128)
            hx = torch.zeros(1,128)
        else:
            cx = cx.data
            hx = hx.data

        value, mu, sigma, (hx, cx) = model(
            (state.unsqueeze(0).float(), (hx, cx))
        )

        # action select
        action_dist = dist.normal.Normal(mu, sigma)
        action = action_dist.sample()

        # env step
        state, reward, done, _ = mujoco.env.step(action.numpy())
        mujoco.env.render()

        total_reward += reward

        done = done or step_len >= cfg.max_step_len

        if done:
            epi_len += 1
            reward_list.append(total_reward)
            print ("Testing over %d episodes, Average reward = %f" % \
                            (epi_len, sum(reward_list)/epi_len,))
            if best_reward < total_reward:
                best_reward = total_reward
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_path,'best' + cfg.env_name+\
                        "."+get_log_name()+"."+str(epi_len)+".pth.tar"))
            if epi_len % cfg.save_frequency == 0:
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_path, cfg.env_name+\
                        "."+get_log_name()+"."+str(epi_len)+".pth.tar"))
            info_str = "Time {}, episode reward {}, episode len {}, step len {}".format(
                            time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                            round(total_reward), epi_len, step_len)
            print(info_str)
            
            wandb.log({'test total_reward' : total_reward})
            wandb.log({'test step_len' : step_len})
        

            total_reward = 0
            step_len = 0
            state = mujoco.env.reset()

            if cfg.mode == 'train':
                time.sleep(1)

        state = torch.from_numpy(state)
        
        




