
import numpy as np
from numpy.random import randint

import gym
from gym import spaces
from gym.utils import seeding

GRID_SIZE = 50
SQ_SIZE = 1. / GRID_SIZE
VW = VH = 300



def random_pos():
    return np.array([randint(0, GRID_SIZE), randint(0, GRID_SIZE)])


class Snake(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(5)
        self.seed()
        self.state = None
        self.viewer = None


    def reset(self):
        self.apple = np.array(random_pos())
        self.snake = np.array([int(GRID_SIZE * .4), int(GRID_SIZE * .6)])
        self.size = 3
        self.tail = np.array([[self.snake[0] - i, self.snake[1]] for i in range(1, self.size + 1)])
        self.theta = 0
        self.vx = 1
        self.vy = 0
        self.state = np.hstack((self.snake, self.vx, self.vy, self.apple))
        return self.state


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        centered = action - 1
        self.theta += centered * (np.pi / 2.)
        self.vx = np.rint(np.cos(self.theta)).astype(np.int64)
        self.vy = np.rint(np.sin(self.theta)).astype(np.int64)
        self.tail = np.vstack((self.snake, self.tail))
        self.snake = np.sum((self.snake, np.array([self.vx, self.vy])), axis=0)

        self.state = np.hstack((self.snake, self.vx, self.vy, self.apple))

        # Check if snake collides with the apple and relocate it.
        rew = int((self.snake == self.apple).all())

        if rew:
            self.apple = random_pos()
            self.size += 1

        if len(self.tail) > self.size:
            self.tail = self.tail[:-1,]

        # Check if snake is still inside the grid and didn't collide with itself.
        done = np.any(self.snake < 0) or np.any(self.snake > GRID_SIZE) \
               or (self.snake == self.tail).all(axis=1).any()

        return self.state, rew, done, {}


    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VW, VH)

        sn = rendering.Transform(translation=self.snake * VW * SQ_SIZE)
        self.viewer.draw_circle(radius=VW * SQ_SIZE / 2., color=(0., .5, 0.)).add_attr(sn)
        ap = rendering.Transform(translation=self.apple * VW * SQ_SIZE)
        self.viewer.draw_circle(radius=VW * SQ_SIZE / 2., color=(.5, 0., 0.)).add_attr(ap)

        for t in self.tail:
            tt = rendering.Transform(translation=t * VW * SQ_SIZE)
            self.viewer.draw_circle(radius=VW * SQ_SIZE / 2., color=(0., .5, 0.)).add_attr(tt)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
