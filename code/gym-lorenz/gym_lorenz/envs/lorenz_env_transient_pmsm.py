#import python libraries
import math

import gym
import os
import csv
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D



class lorenzEnv_transient(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ):

        self.input_min = -2
        self.input_max = 2
        self.state_dim = 6  # 包含两个HR系统状态差分量
        self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(2,), dtype=np.float32)
        self.state = None#状态变量六维
        self.state0=None#六维
        self.state1 = None#三维
        self.state2 = None#六维
        self.dis=0
        self.u1 = 0
        self.u2 = 0
        self.u3 = 0
        self.t=0
        self.u = 10
        self.i = 28
        self.o = 8 / 3
        self.a=5.46
        self.b=20

    def reset(self):
        # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
        state1=np.random.uniform(low=-10, high=10, size=(3,))
        state2=np.random.uniform(low=-10, high=10, size=(3,))
        self.state1=state1
        self.state2 = state2
        dxdt_controlled = -state1[0] + state1[1] * state1[2]
        dydt_controlled = -state1[1] -state1[0] * state1[2] + self.b * state1[2]
        dzdt_controlled = self.a * (state1[1] - state1[2])
        # dxdt_controlled = self.u * (state1[1] - state1[0])
        # dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
        # dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
        dxdt_controlled_2 = -state2[0] + state2[1] * state2[2]
        dydt_controlled_2 = -state2[1] - state2[0] * state2[2] + self.b * state2[2]
        dzdt_controlled_2 = self.a * (state2[1] - state2[2])
        self.state0 = [state1[0],state1[1],state1[2],dxdt_controlled,dydt_controlled,dzdt_controlled]
        #改到这环境从三维改成六维
        self.state2 = np.array([state2[0],state2[1],state2[2],dxdt_controlled_2,dydt_controlled_2,dzdt_controlled_2])
        self.t = 0
        return self._get_observation()

    def _get_observation(self):
        return np.array(self.state0) - np.array(self.state2)

    def _get_current(self):
        return [self.state1[0], self.state2[0]]

    def _get_current1(self):
        return [self.state1[1], self.state2[1]]

    def _get_current2(self):
        return [self.state1[2], self.state2[2]]

    def step(self, action):
        # 将控制输入应用到受控系统，并模拟动态过程
        self.u1 = np.clip(action[0], self.input_min, self.input_max)
        self.u2 = np.clip(action[1], self.input_min, self.input_max)
        # self.u3 = np.clip(action[2], self.input_min, self.input_max)
        # self.u4 = np.clip(action[3], self.input_min, self.input_max)
        self.state = self._get_observation()
        # 计算奖励函数
        state1 = self.state1
        state2=self.state2
        self.target_system_noise = np.random.normal(loc=0,scale=3, size=(3,))
        dxdt_controlled = -state1[0] + state1[1] * state1[2]
        dydt_controlled = -state1[1] - state1[0] * state1[2]+ self.b * state1[2]
        dzdt_controlled = self.a * (state1[1] - state1[2])

        dxdt_controlled_2 = -state2[0] + state2[1] * state2[2]+self.u1*20+self.target_system_noise[0]
        dydt_controlled_2 = -state2[1] - state2[0] * state2[2] + self.b * state2[2]+self.u2*20+self.target_system_noise[1]
        dzdt_controlled_2 = self.a * (state2[1] - state2[2])+self.target_system_noise[2]
        # dxdt_controlled = self.u * (state1[1] - state1[0])+self.target_system_noise[0]
        # dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]+self.target_system_noise[1]
        # dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]+self.target_system_noise[2]
        state1[0]=state1[0]+dxdt_controlled*0.01
        state1[1] = state1[1] + dydt_controlled * 0.01
        state1[2] = state1[2] + dzdt_controlled * 0.01

        state2[0] = state2[0] + dxdt_controlled_2 * 0.01
        state2[1] = state2[1] + dydt_controlled_2 * 0.01
        state2[2] = state2[2] + dzdt_controlled_2 * 0.01

        self.state1=state1
        # dxdt_controlled = self.u * (state1[1] - state1[0])
        # dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
        # dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
        dxdt_controlled = -state1[0] + state1[1] * state1[2]
        dydt_controlled = -state1[1] - state1[0] * state1[2] + self.b * state1[2]
        dzdt_controlled = self.a * (state1[1] - state1[2])
        self.state0 = [state1[0],state1[1],state1[2],dxdt_controlled, dydt_controlled, dzdt_controlled]

        dxdt_controlled_2 = -state2[0] + state2[1] * state2[2]
        dydt_controlled_2 = -state2[1] - state2[0] * state2[2] + self.b * state2[2]
        dzdt_controlled_2 = self.a * (state2[1] - state2[2])

        self.state2 = np.array([state2[0], state2[1], state2[2], dxdt_controlled_2, dydt_controlled_2, dzdt_controlled_2])
        # 更新环境状态
        self.state = self.state0 - self.state2
        now=self._get_observation()
        reward = -sum(abs(x) for x in now[0:3])-(sum(abs(x) for x in now[0:3]))**(1/10) # 使用简单的欧氏距离作为奖励
        # reward = -sum(abs(x) for x in now[0:3])  # 使用简单的欧氏距离作为奖励
        self.t=self.t+0.01
        # if abs(self.u1-self.input_min)<10 or abs(self.u1-self.input_max)<10:
        #     reward=reward-1
        # if abs(self.u2-self.input_min)<10 or abs(self.u2-self.input_max)<10:
        #     reward=reward-1
        if self.t ==5 or reward<-1e6:
            done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
        else:
            done = False
        return self._get_observation(), reward, done, {}


    def render(self, mode='human'):
        pass  # 可以添加可视化功能在这里




