"""
For this script to run the following hardware is needed:
* Crazyflie 2.0
* Crazyradio PA
* Flow deck
* Optitrack system
"""
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

# utils
import os
import sys
import time
import pickle
import logging
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils.util import computeExperimentID, to_gym_interface

# crazyflie
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

# optitrack
from utils.optitrack import NatNetClient

# URI for the crazyflie to be connected
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EC')

# only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# plot logger
logger = {}
logger['pos'] = []

# position got from optitrack (global variable to be rewritten by optitrack callback)
position_optitrack = np.zeros((3,))

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
    if id == 2:
        # optitrack z x y
        # crazyflie x y z
        position_optitrack[0] = position[2]
        position_optitrack[1] = position[0]
        position_optitrack[2] = position[1]
        #print(position_optitrack)
        #logger["pos"].append(pos_ot)
        logger["pos"].append([position[2], position[0], position[1]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, help='Change the model loading directory here')
    args = parser.parse_args()

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
            goal_reached = False
            while not goal_reached:
                # TODO: output velocity
                velocity_x, velocity_y = 0.5, 0
                if position_optitrack[1] > 0.0:
                    goal_reached = True
                    
                # calling motion_commander._thread.set_vel_setpoint()
                motion_commander.start_linear_motion(velocity_x, velocity_y, 0.0)
                time.sleep(0.1)
        pickle.dump(logger, open("experimentsCrazy/{}/logger.pkl".format(args.dir), 'wb'))
        print("Navigation completed!")