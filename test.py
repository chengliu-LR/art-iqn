import os
import sys
import yaml
import time
import pickle
import argparse
import logging
import gym

import torch
import numpy as np
from collections import deque

from agent import DQNAgent
from utils.util import to_gym_interface
import matplotlib.pyplot as plt

import crazyflie_env
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import FullState

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, help='Change the model loading directory here')
    parser.add_argument('--env', default='CrazyflieEnv-v0', help='Training environment')
    parser.add_argument('--num_directions', default=8, type=int, help='Discrete directions')
    parser.add_argument('--num_speeds', default=1, type=int, help='Discrete velocities')
    parser.add_argument('--max_velocity', default=1.0, type=float, help='Maximum velocity')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--cvar', default=0.2, type=float, help="Give the quantile value of the CVaR tail")
    parser.add_argument('--seed', default=5, help=" Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=256, type=int, help='Hidden layer size of neural network')
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size of the replay memory')
    args = parser.parse_args()

    env = gym.make("CrazyflieEnv-v0")
    state = env.reset()
    print('initial state:', state) #observable state: px, py, vx, vy, radius
    state_size = len(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eps=0.0 # no exploration during test

    agent = DQNAgent(state_size=state_size,
                        num_directions=args.num_directions,
                        num_speeds=args.num_speeds,
                        layer_size=args.layer_size,
                        n_step=args.n_step,
                        BATCH_SIZE=args.batch_size,
                        BUFFER_SIZE=args.buffer_size,
                        LR=args.lr, 
                        TAU=args.tau,
                        GAMMA=args.gamma,
                        UPDATE_EVERY=args.update_every,
                        device=device,
                        seed=args.seed,
                        distortion=args.distortion,
                        con_val_at_risk=args.cvar)
    
    # load trained model
    dir = './experimentsCrazy/{}/IQN.pth'.format(args.dir)
    agent.qnetwork_local.load_state_dict(torch.load(dir))
    agent.action_space = agent.build_action_space(args.max_velocity)

    done = False
    score = 0
    while not done:
        action_id, action = agent.act(to_gym_interface(state), eps)
        next_state, reward, done, info = env.step(action)
        #agent.update(state, action, reward, next_state, done)
        state = next_state
        score += reward
        print("x {}, y {}, reward {}, info {} vx {}, vy {}".format(round(state.position[0], 5), round(state.position[1], 5), round(reward, 2), info, round(action.vx, 2), round(action.vy, 2)))
    print("Episodic return:", score)

    env.render(mode='video', output_file="./figures/iqn_random_init.gif")