import gym
import torch
import numpy as np

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss
    
            
def eval_runs(agent, eps, frame, writer):
    """
    Makes an evaluation run with the current epsilon
    """
    env = gym.make("CartPole-v0")
    reward_batch = []
    for i in range(5):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step(action)
            env.render()
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    writer.add_scalar("Reward", np.mean(reward_batch), frame)