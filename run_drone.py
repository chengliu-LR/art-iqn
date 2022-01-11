import os
import sys
import yaml
import time
import pickle
import argparse

from drone import DroneEnv, DroneEnv2D
import torch
import numpy as np
from collections import deque

from agent import DQNAgent
from utils import eval_runs, computeExperimentID

def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01):
    """Deep Q-Learning
    Params
    ======
        eps_fixed (int): whether epsilon greedy exploration
        min_eps (float)L minimum epsilon greedy exploration rate
        n_episodes (int): maximum number of training episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []

    # logger
    logger = {}
    logger['scores'] = []
    logger['scores_window'] = []

    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1

    eps_start = 1
    i_episode = 1

    state = env.reset()
    score = 0
    
    for frame in range(1, frames+1):
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        score += reward

        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame * (1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps * ((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        # if frame % 5000 == 0:
        #     eval_runs(agent, eps, frame)
        
        if done:
            scores_window.append(score)       
            logger['scores'].append(score)
            logger['scores_window'].append(np.mean(scores_window))
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode += 1
            state = env.reset()
            score = 0
    
    pickle.dump(logger, open("{}/{}/logger.pkl".format(args.save_dir, currentExperimentID), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='experiments_drone2D', help='Change the experiment saving directory here')
    parser.add_argument('--env', default='CartPole-v0', help='Training environment')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--cvar', default=0.2, type=float, help="Give the quantile value of the CVaR tail")
    parser.add_argument('--seed', default=5, help=" Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=256, type=int, help='Hidden layer size of neural network')
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size of the replay memory')
    parser.add_argument('--frames', default=100000, type=int, help='Number of training frames')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    currentExperimentID = computeExperimentID(args.save_dir)
    print("Current experiment ID: {}".format(currentExperimentID))
    os.mkdir("{}/{}/".format(args.save_dir, currentExperimentID))

    with open("{}/{}/arguments".format(args.save_dir, currentExperimentID), 'w') as f:
        yaml.dump(args.__dict__, f)

    with open("{}/{}/{}".format(args.save_dir, currentExperimentID, "".join(sys.argv)), 'w') as f:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    np.random.seed(args.seed)
    env = DroneEnv2D()
    state_size = env.state_size
    action_size = env.action_size

    agent = DQNAgent(state_size=state_size,
                        action_size=action_size,
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

    # set epsilon frames to 0 so no epsilon exploration
    eps_fixed = False

    # logger for multiple plots

    t_start = time.time()
    run(frames = args.frames, eps_fixed=eps_fixed, eps_frames=5000, min_eps=0.025)
    t_end = time.time()
    
    print("\nTraining time: {}min".format(round((t_end-t_start) / 60, 2)))
    torch.save(agent.qnetwork_local.state_dict(), "{}/{}/IQN.pth".format(args.save_dir, currentExperimentID))