import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import math
import time

import gym
# from utils import *
from config import Config as cfg
from env import Mujoco
from model import ActorCritic
from train import train
from test import test

import wandb
import os
os.environ["WANDB_API_KEY"] = "01cfeb4ac3554c7135be197ea77228ae1ef20f98"
wandb.login()

def get_log_name():
    now = time.localtime()
    return str(now.tm_mon) +'_' + str(now.tm_mday)+ '_' + str(now.tm_hour) + '_'+ str(now.tm_min)

if __name__ == '__main__':
    exp_time = get_log_name()
    # 시드 고정

    # env setting
    mjk = Mujoco(cfg.env_name)

    # shared model 설정
    shared_model = ActorCritic(mjk.env.observation_space.shape[0], mjk.env.action_space.shape[0])
    shared_model.share_memory()
    print( shared_model.share_memory())

    # optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    # optimizer.share_memory()

    # process마다 train
    processes = []
    # test하는 process 하나: rank = process_num
    p = mp.Process(target=test, args=(cfg.num_processes, cfg, shared_model, exp_time))
    p.start()
    processes.append(p)
    for rank in range(0, cfg.num_processes):
        # process 개수 만큼 train: rank = 0 ~ (process_num-1)
        p = mp.Process(target=train, args=(rank, cfg, shared_model, exp_time))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

    

