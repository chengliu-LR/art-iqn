import numpy as np
import math
class DroneEnv():
    """
    Description:
        A 1-D Drone toy environment to test risk-averse policy.
    Observation:
        [x, v]
        Ind     Observation
        0       Drone Position
        1       Drone Velocity
    Actions:
        Type: Integer
        Ind   Action
        1     Accelerate
        0     Constant Velocity
        Note: The 1-D Drone environment implemented in the O-RAAC paper is continuous,
        Here we modify it to discrete to combine with IQN algorithm.
    Reward:
        Penalizes episode length, exceeding speed limit while give positive reward for goal acievement.
    Starting State:
        Starting at [0.0, 0.0]
    Episode Termination:
        Episode length is greater than 400;
        When the agent reaches the goal.
    """

    def __init__(self):
        self.position = 0.0
        self.velocity = 0.0
        self.dt = 0.1
        self.goal_position = 2.5
        self.state_size = np.zeros(2,).shape
        self.action_size = 2

    def step(self, action):
        #self.position = self.position + self.velocity * self.dt + 0.5 * action * (self.dt ** 2)
        self.position = self.position + self.velocity * self.dt
        self.velocity = self.velocity + action * self.dt
        reward = -10.0

        speed_limit_exceeded = bool(self.velocity > 1.0)
        if speed_limit_exceeded:
            reward += -25.0 * np.random.binomial(1, 0.4, 1)[0]

        done = bool(self.position >= self.goal_position)
        if done:
            reward += 370.0
        return np.array((self.position, self.velocity), dtype=np.float32), reward, done, {}


    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        return np.array((self.position, self.velocity), dtype=np.float32)


class DroneEnv2D():
    """
    Description:
        A 2-D Drone toy environment to test risk-averse policy.
    Observation:
        [x, y, vx, vy]
        Ind     Observation
        0       Drone x position
        1       Drone y position
        2       Drone velocity along x axis
        3       Drone velocity along y axis
    Actions:
        Type: Integer
        Ind   Action
        0     Constant Velocity
        1     Accelerate along x axis
        2     Accelerate along y axis
    Reward:
        Penalizes episode length, exceeding speed limit while give positive reward for goal acievement.
    Starting State:
        Starting at [0.0, 0.0, 0.0, 0.0]
    Episode Termination:
        Episode length is greater than 400;
        When the agent reaches the goal area.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.accel_rate = 1.0
        self.dt = 0.1
        self.goal_position = 2.5
        self.state_size = np.zeros(4,).shape
        self.action_size = 3
    
    def step(self, action):
        self.x = self.x + self.vx * self.dt
        self.y = self.y + self.vy * self.dt
        self.vx = self.vx + self.accel_rate * self.dt if action == 1 else self.vx
        self.vy = self.vy + self.accel_rate * self.dt if action == 2 else self.vy
        reward = -10.0

        speed_limit_exceeded = bool((self.vx ** 2 + self.vy ** 2) > 2.0)
        #TO-DO: check each velocity separately
        if speed_limit_exceeded:
            reward += -25.0 * np.random.binomial(1, 0.2, 1)[0]
        
        done = bool(self.x >= self.goal_position and
                    self.y >= self.goal_position)
        if done:
            reward += 370
        return np.array((self.x, self.y, self.vx, self.vy), dtype=np.float32), reward, done, {}
    
    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        return np.array((self.x, self.y, self.vx, self.vy), dtype=np.float32)