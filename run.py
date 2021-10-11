import gym
import time
import torch
import numpy as np
from agent import DQNAgent
from utils import eval_runs
from collections import deque
from torch.utils.tensorboard import SummaryWriter

def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01):
    """Deep Q-Learning
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []
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
        agent.step(state, action, reward, next_state, done, writer)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % 5000 == 0:
            eval_runs(agent, eps, frame, writer)
        
        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame)
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode +=1 
            state = env.reset()
            score = 0              

    return output_history


if __name__ == "__main__":

    writer = SummaryWriter("runs/"+"IQN_CP_5")
    seed            = 5
    n_step          = 1
    UPDATE_EVERY    = 1
    BATCH_SIZE      = 8
    GAMMA           = 0.99
    TAU             = 1e-2
    LR              = 1e-3
    BUFFER_SIZE     = 100000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    env = gym.make("CartPole-v0")

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    agent = DQNAgent(state_size=state_size,    
                        action_size=action_size,
                        layer_size=256,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY, 
                        device=device,
                        seed=seed)


    # set epsilon frames to 0 so no epsilon exploration
    eps_fixed = False

    t0 = time.time()
    final_average100 = run(frames = 60000, eps_fixed=eps_fixed, eps_frames=5000, min_eps=0.025)
    t1 = time.time()
    
    print("Training time: {}min".format(round((t1-t0)/60,2)))
    torch.save(agent.qnetwork_local.state_dict(), "IQN.pth")
