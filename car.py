import numpy as np

class CarEnv():
    """
    Description:
        A 1-D car toy environment to test risk-averse policy.
    Observation:
        [x, v]
        Ind     Observation
        0       Car Position
        1       Car Velocity
    Actions:
        Type: Integer
        Ind   Action
        1     Accelerate
        0     Constant Velocity
        -1    Decelerate
        Note: The 1-D car environment implemented in the O-RAAC paper is continuous,
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
        self.action_size = 3

    def step(self, action):
        self.position = self.position + self.velocity * self.dt + 0.5 * action * (self.dt ** 2)
        self.velocity = self.velocity + action * self.dt

        reward = -10.0

        speed_limit_exceeded = bool(self.velocity >= 1.0)
        if speed_limit_exceeded:
            reward += -25.0 * np.random.binomial(1,0.4,1)[0]

        done = bool(self.position >= self.goal_position)
        if done:
            reward += 370.0

        return np.array((self.position, self.velocity), dtype=np.float32), reward, done, {}


    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        return np.array((self.position, self.velocity), dtype=np.float32)