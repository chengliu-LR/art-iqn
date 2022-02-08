import ast
import logging
import math
import sys
import time
import pickle
from threading import Timer

import numpy as np
from utils.optitrack import NatNetClient

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper


uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EC')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

logger = {}
logger["pos"] = []

pos_ot = np.zeros((1, 3))
vex, vey, vez = 0.0, 0.0, 0.0


class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """
    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        self._lg_stab.add_variable('stateEstimate.vx', 'float')
        self._lg_stab.add_variable('stateEstimate.vy', 'float')
        self._lg_stab.add_variable('stateEstimate.vz', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(300, self._cf.close_link)  # adjust it !!!!
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        global vex, vey, vez
        vex = data['stateEstimate.vx']
        vey = data['stateEstimate.vy']
        vez = data['stateEstimate.vz']

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False


# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
def receiveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount,
                    skeletonCount, labeledMarkerCount, latency, timecode, timecodeSub, 
                    timestamp, isRecording, trackedModelsChanged):
    pass

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame


def receiveRigidBodyFrame(id, position, rotation):
    #global pos_ot, logger
    if id == 1:
        # opti_track z x y
        # crazyflie x y z
        pos_ot[0, 0] = position[2]
        pos_ot[0, 1] = position[0]
        pos_ot[0, 2] = position[1]
        print(pos_ot)
        #logger["pos"].append(pos_ot)
        logger["pos"].append([position[2], position[0], position[1]])


if __name__ == '__main__':

    streamingClient = NatNetClient()  # Create a new NatNet client
    streamingClient.newFrameListener = receiveNewFrame
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run()  # Run perpetually on a separate thread.

 # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    le = LoggingExample(uri)

    file = open('./dat00.csv', 'w')
    file.write('timeStamp, OTx, OTy, OTz, vex, vey, vez\n')

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    time.sleep(5)

    if (pos_ot[0, 0] == 0) & ((pos_ot[0, 1] == 0) & (pos_ot[0, 2] == 0)):
        raise ImportError('No connection')  # if OptiTrack is not connected

    #while le.is_connected:
    time0 = round(time.time()*1000) % 1000000
    try:
        time.sleep(0.01)
        time_now = round(time.time()*1000) % 1000000-time0  # timestamp (ms)
        file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
            time_now, pos_ot[0, 0], pos_ot[0, 1], pos_ot[0, 2], vex, vey, vez))
        # le._cf.commander.send_setpoint(roll, pitch, yaw, thrust) # thrust 0-FFFF

        for y in range(10):
            le._cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            time.sleep(0.1)

        for _ in range(20):
            le._cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)

        for _ in range(50):
            le._cf.commander.send_hover_setpoint(0.5, 0, 0, 0, 0.4)

        # comment this for only landing by pressing ctrl+z
        # for y in range(10):
        #     le._cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
        #     time.sleep(0.1)

        le._cf.commander.send_stop_setpoint()
        pickle.dump(logger, open("./logger.pkl", "wb"))

    except KeyboardInterrupt:
        print('stop')
        le._cf.commander.send_stop_setpoint()
        raise

