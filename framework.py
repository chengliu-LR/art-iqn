"""
For this script to run the following hardware is needed:
* Crazyflie 2.0
* Crazyradio PA
* Flow deck
* Optitrack system
! Always check the user ID before running this script!
"""
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from agent import DQNAgent

# utils
import os
import gym
import sys
import time
import pickle
import logging
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils.util import to_gym_interface_pos

# crazyflie
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

# optitrack
from utils.optitrack import NatNetClient

# iqn agent
import crazyflie_env
from crazyflie_env.envs.utils.state import ObservableState

# URI for the crazyflie to be connected
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EC')

# only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# plot logger
logger = {}
logger['pos'] = []

# position got from optitrack (global variable to be rewritten by optitrack callback)
pos_opti = np.zeros((3,))

def receiveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount,
                    skeletonCount, labeledMarkerCount, latency, timecode, timecodeSub, 
                    timestamp, isRecording, trackedModelsChanged):
    """
    A callback function that gets connected to the optitrack NatNet client.
    Called once per mocap frame.
    """
    pass

def receiveRigidBodyFrame(id, position, rotation):
    """
    A callback function that gets connected to the NatNet client.
    Called once per rigid body per frame.
    """
    #global pos_ot, logger
    if id == 1:
        # optitrack z x y
        # crazyflie x y z
        pos_opti[0] = position[2]
        pos_opti[1] = position[0]
        pos_opti[2] = position[1]
        #print(pos_opti)
        #logger["pos"].append(pos_ot)
        logger["pos"].append([position[2], position[0], position[1]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=176, help='Change the model loading directory here')
    parser.add_argument('--env', default='CrazyflieEnv-v0', help='Training environment')
    parser.add_argument('--num_directions', default=4, type=int, help='Discrete directions')
    parser.add_argument('--num_speeds', default=3, type=int, help='Discrete velocities')
    parser.add_argument('--max_velocity', default=1.0, type=float, help='Maximum velocity')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--cvar', default=0.2, type=float, help="Give the quantile value of the CVaR tail")
    parser.add_argument('--seed', default=5, help=" Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=512, type=int, help='Hidden layer size of neural network')
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size of the replay memory')
    parser.add_argument('--test_eps', default=0.0, type=float, help='Learning rate')
    args = parser.parse_args()

    env = gym.make("CrazyflieEnv-v0")
    env.set_obstacle_num(0)
    state = env.reset()
    state_size = len(to_gym_interface_pos(state))
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
                        con_val_at_risk=args.cvar)
    
    # load trained model
    dir = './experimentsCrazy/{}/IQN.pth'.format(args.dir)
    agent.qnetwork_local.load_state_dict(torch.load(dir))
    agent.action_space = agent.build_action_space(args.max_velocity)


    # start receiving optitrack frames
    streaming = NatNetClient()
    streaming.newFrameListener = receiveNewFrame
    streaming.rigidBodyListener = receiveRigidBodyFrame
    streaming.run()

    #agent = pickle.load(open("experimentCrazy/{}/agent.pkl".format(args.dir), 'rb'))
    
    # initialize the low-level crazyflie drivers
    cflib.crtp.init_drivers()

    # start crazyflie
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf) as motion_commander:
            with Multiranger(scf) as multiranger:
                goal_reached = False
                radius = 0.05
                gx, gy = 0.0, 2.0
                orientation = 0.0
                goal_position = np.array((gx, gy))

                while not goal_reached:
                    robot_position = np.array((pos_opti[0], pos_opti[1]))
                    goal_distance = np.linalg.norm(robot_position - goal_position)
                    #ranger_raw = [multiranger.front, multiranger.back, multiranger.left, multiranger.right]
                    ranger_raw = [multiranger.front, multiranger.left, multiranger.back, multiranger.right]
                    ranger_reflections = np.array([ranger_raw[i] if ranger_raw[i] is not None else 3.0 for i in range(4)])
                    clipped_reflections = np.clip(ranger_reflections, 0.0, 3.0)
                    state = ObservableState(robot_position, goal_distance, orientation, clipped_reflections)
                    print(robot_position, clipped_reflections)
                    action_id, action = agent.act(to_gym_interface_pos(state), eps)
                    motion_commander.start_linear_motion(action.vx, action.vy, 0.0)
                    time.sleep(0.1)
                    if goal_distance < 20 * radius:
                        goal_reached = True

                # calling motion_commander._thread.set_vel_setpoint()
        pickle.dump(logger, open("experimentsCrazy/{}/trajectory_closer_flag.pkl".format(args.dir), 'wb'))
        print("Navigation completed!")