import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class QTankEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.A = [28, 32, 28, 32]  # (cm^2) cross-section of Tank i
        self.a = [0.071, 0.057, 0.071, 0.057]  # (cm^2) cross-section of the outlet hole
        self.kc = 1.0  # (V/cm) with kc=1, level outputs are in cm
        self.g = 981  # (cm/s^2)

        self.gamma = [0.7, 0.6]  # parameter (adjust the 3-way valve)

        self.h_max = 20  # (cm) the hight of each tank
        self.q_max = 2.5  # (l/min) pumps capacity

        self.k = 1.0  # considering that the control signal v
                             # corresponds to the desired flow (q=k*v)

        self.observation_space = spaces.Box(low=0, high=self.h_max/self.kc, shape=(4,))

        self.q_max_converted = self.q_max*1000/60  # (cm^3/s)
        self.action_space = spaces.Box(low=0, high=self.q_max_converted/self.k, shape=(2,))

        self.state = None
        self.dt = 0.1  # integration step (seconds)
        self.steps_beyond_done = None

        self.seed()

        self.info = {'method': 'Euler', 'step': self.dt}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # states: h1, h2, h3, h4
        self.state = self.np_random.uniform(low=2, high=18, size=(4))
        return np.array(self.kc*self.state)

    def step(self, u):

        h1, h2, h3, h4 = self.state  # h := level
        a1, a2, a3, a4 = self.a
        A1, A2, A3, A4 = self.A
        gamma1, gamma2 = self.gamma
        k1 = k2 = self.k
        g = self.g

        # system hdot = f(h,u)
        h1dot = -(a1/A1)*np.sqrt(2*g*h1) + (a3/A1)*np.sqrt(2*g*h3) + gamma1*k1*u[0]/A1
        h2dot = -(a2/A2)*np.sqrt(2*g*h2) + (a4/A2)*np.sqrt(2*g*h4) + gamma2*k2*u[1]/A2
        h3dot = -(a3/A3)*np.sqrt(2*g*h3) + (1-gamma2)*k2*u[1]/A3
        h4dot = -(a4/A4)*np.sqrt(2*g*h4) + (1-gamma1)*k1*u[0]/A4
        # hdot = [h1dot, h2dot, h3dot, h4dot]

        # Euler: hnew = I*h + hdot*dt
        h1 = h1 + self.dt * h1dot
        h2 = h2 + self.dt * h2dot
        h3 = h3 + self.dt * h3dot
        h4 = h4 + self.dt * h4dot

        h = np.array([h1, h2, h3, h4])
        y = self.kc * h

        self.state = h

        done = any(h <= 0) \
            or any(h >= self.h_max)
        done = bool(done)

        # reward: =1 if inside limits, =0 if not
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # any tank out of limits
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this \
                  environment has already returned done = True. You \
                  should always call 'reset()' once you receive \
                  'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return y, reward, done, self.info

    '''
    def render(self, mode='human'):
      ...

    def close(self):
      ...
    '''
