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
from utils.util import to_gym_interface_pos
import matplotlib.pyplot as plt

import crazyflie_env
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import FullState

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, help='Change the model loading directory here')
    parser.add_argument('--env', default='CrazyflieEnv-v0', help='Training environment')
    parser.add_argument('--num_directions', default=4, type=int, help='Discrete directions')
    parser.add_argument('--num_speeds', default=1, type=int, help='Discrete velocities')
    parser.add_argument('--max_velocity', default=1.0, type=float, help='Maximum velocity')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--sample_cvar', default=1, type=float, help="Enable cvar value sampling from the uniform distribution")
    parser.add_argument('--cvar', default=0.2, type=float, help="Give the quantile value of the CVaR tail")
    parser.add_argument('--seed', default=5, type=int, help="Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=512, type=int, help='Hidden layer size of neural network')
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size of the replay memory')
    parser.add_argument('--render_mode', default='trajectory', help='Render mode')
    parser.add_argument('--render_density', default=8, type=int, help='Render density')
    parser.add_argument('--obstacle_num', default=3, type=int, help='Number of obstacles set in the env, only functions when random_obstacle is True')
    parser.add_argument('--test_eps', default=0.0, type=float, help='Learning rate')
    parser.add_argument('--random_obstacle', default=1, type=int, help='Enable random obstacle generation or fixed obstacle position')
    parser.add_argument('--eval', default=1, type=int, help='enable crazyflie env evaluation mode to fix obstacle numbers to obstacle_num')
    args = parser.parse_args()

    env = gym.make("CrazyflieEnv-v0")
    env.enable_random_obstacle(args.random_obstacle)
    env.set_obstacle_num(args.obstacle_num)
    env.enable_eval_mode(args.eval)
    state, obs_num = env.reset()
    state_size = len(to_gym_interface_pos(state))

    # if you want to set robot initial position by hand:
    #env.robot.set_state(args.init_x, args.init_y, 0, 2.5, 0, 0, env.obstacle_segments) # generalize to a slightly modified goal position
    #state = env.robot.observe()
    print('initial state:', state) #observable state: px, py, vx, vy, radius

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eps=args.test_eps # exploration rate during test

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
                        con_val_at_risk=bool(args.sample_cvar),
                        eval = bool(args.eval))
    
    # load trained model
    dir = './experimentsCrazy/{}/IQN.pth'.format(args.dir)
    agent.qnetwork_local.load_state_dict(torch.load(dir))
    agent.action_space = agent.build_action_space(args.max_velocity)

    # evaluation collision rate, success rate, task finishing time, danger zone
    cvar_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    densities = [1, 3, 5, 7]
    num_eval_episodes = 100
    seeds = [i for i in range(num_eval_episodes)]

    logger_eval = open("./experimentsCrazy/{}/eval.csv".format(args.dir), 'w+')
    logger_eval.write("Density, CVaR, Success, Timeout, Collision, Time, Danger, Episodic Scores\n")
    logger_eval.flush()

    for density in densities:
        for cvar in cvar_values:
            env.set_obstacle_num(density)
            episodic_scores = []
            episodic_finish_time = []
            danger_steps = []
            success = 0
            collision = 0
            timeout = 0
            for epoch in range(num_eval_episodes):
                np.random.seed(seeds[epoch])
                state, obs_num = env.reset()
                if epoch <= 2:
                    print("check if seed works:", cvar, obs_num, env.robot.get_goal_position())
                done = False
                score = 0
                danger = 0
                while not done:
                    action_id, action = agent.act(to_gym_interface_pos(state), eps, cvar)
                    next_state, reward, done, info = env.step(action)
                    #agent.update(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    #print("dist {}, reward {}, {} vx {}, vy {} ranger {}".format(round(state.goal_distance, 6), round(reward, 2), info, round(action.vx, 2), round(action.vy, 2), state.ranger_reflections[1]))
                    if info == 'Danger':
                        danger += 1
                    if done:
                        if info == 'Reached':
                            success += 1
                            episodic_finish_time.append(env.global_time)
                        elif info == 'Collision':
                            collision += 1
                        elif info == 'Timeout':
                            timeout += 1
                        episodic_scores.append(score)
                        danger_steps.append(danger)
            #print(cvar, success / num_eval_episodes, collision / num_eval_episodes, np.average(episodic_finish_time), np.average(danger_steps))
            logger_eval.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(density, cvar, success / num_eval_episodes, timeout / num_eval_episodes, 
                            collision / num_eval_episodes, np.average(episodic_finish_time), np.average(danger_steps), np.average(episodic_scores)))
            logger_eval.flush()