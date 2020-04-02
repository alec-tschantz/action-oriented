import numpy as np
from core.config import *


class Environment(object):

    def __init__(self,
                 env_size=ENVIRONMENT_SIZE,
                 init_distance=INIT_DISTANCE,
                 source_size=SOURCE_SIZE,
                 agent_size=AGENT_SIZE,
                 velocity=VELOCITY):

        self.env_size = env_size
        self.init_distance = init_distance
        self.source_size = source_size
        self.agent_size = agent_size
        self.vel = velocity

        self.pos = None
        self.s_pos = None
        self.theta = None
        self.reset()

    def reset(self):
        rand_loc = np.random.rand() * (2 * np.pi)
        fx = self.env_size / 2 + (self.init_distance * np.cos(rand_loc))
        fy = self.env_size / 2 + (self.init_distance * np.sin(rand_loc))

        self.pos = [fx, fy]
        self.s_pos = [self.env_size / 2, self.env_size / 2]
        self.theta = np.random.rand() * (2 * np.pi)
        self.observe()

    def observe(self):
        fx = self.pos[0] + (self.agent_size * np.cos(self.theta))
        fy = self.pos[1] + (self.agent_size * np.sin(self.theta))
        f_dis = self.dis(fx, fy, self.s_pos[0], self.s_pos[1])
        b_dis = self.dis(self.pos[0], self.pos[1],
                         self.s_pos[0], self.s_pos[1])
        if f_dis > b_dis:
            o = NEG_GRADIENT
        else:
            o = POS_GRADIENT
        return o

    def act(self, a):
        if a == RUN and self.distance() > self.source_size:
            self.pos[0] += (self.vel * np.cos(self.theta))
            self.pos[1] += (self.vel * np.sin(self.theta))
            self.check_bounds()
        elif a == TUMBLE:
            self.theta = np.random.rand() * (2 * np.pi)

        return self.observe()

    def distance(self):
        return self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])

    def check_bounds(self):
        if self.pos[0] > self.env_size:
            self.pos[0] = self.env_size
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] > self.env_size:
            self.pos[1] = self.env_size
        if self.pos[1] < 0:
            self.pos[1] = 0

    @staticmethod
    def dis(x1, y1, x2, y2):
        return np.sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))
