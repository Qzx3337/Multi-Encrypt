#import python libraries
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



# class lorenzEnv_transient(gym.Env):
#     # #一个系统的稳定
#     # metadata = {'render.modes': ['human', 'rgb_array']}
#     #
#     # def __init__(self, ):
#     #
#     #     self.input_min = -2
#     #     self.input_max = 2
#     #     self.state_dim = 8
#     #     self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
#     #     self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
#     #     self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(2,), dtype=np.float32)
#     #     self.state = None
#     #     self.state0=None
#     #     self.state1 = None
#     #     self.state2 = None
#     #     self.dis=0
#     #     self.u1 = 0
#     #     self.u2 = 0
#     #     self.u3 = 0
#     #     self.t=0
#     #     self.a=30
#     #     self.b=1
#     #     self.c=36
#     #     self.d=0.5
#     #     self.h = 0.003
#     #
#     # def reset(self):
#     #     # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
#     #     state1=np.random.uniform(low=0, high=5, size=(4,))
#     #     #state1=np.array([2,3.01,4.01,0])
#     #     self.state1=state1
#     #     dx_1_controlled = self.a*(2*state1[3]*state1[3]*(state1[1]-state1[0])+self.d*state1[0])
#     #     dx_2_controlled = self.b*(2*state1[3]*state1[3]*(state1[0]-state1[1])-state1[2])
#     #     dx_3_controlled = self.c*(state1[1]-self.h*state1[2])
#     #     dx_4_controlled = state1[1]-state1[0]-0.01*state1[3]
#     #     self.state0 = [state1[0],state1[1],state1[2],state1[3],dx_1_controlled,dx_2_controlled,dx_3_controlled,dx_4_controlled]
#     #     #改到这环境从三维改成六维
#     #     self.state2 = np.array([0, 0, 0, 0, 0, 0,0,0])
#     #     self.t = 0
#     #     return self._get_observation()
#     #
#     # def _get_observation(self):
#     #     return np.array(self.state0) - np.array(self.state2)
#     #
#     # def _get_current(self):
#     #     return [self.state1[0], self.state2[0]]
#     #
#     # def _get_current1(self):
#     #     return [self.state1[1], self.state2[1]]
#     #
#     # def _get_current2(self):
#     #     return [self.state1[2], self.state2[2]]
#     #
#     # def step(self, action):
#     #     # 将控制输入应用到受控系统，并模拟动态过程
#     #     self.u1 = np.clip(action[0], self.input_min, self.input_max)
#     #     self.u2 = np.clip(action[1], self.input_min, self.input_max)
#     #     #self.u3 = np.clip(action[2], self.input_min, self.input_max)
#     #     self.state = self._get_observation()
#     #     # 计算奖励函数
#     #     state1 = self.state1
#     #     #self.target_system_noise = np.random.normal(loc=0,scale=3, size=(3,))
#     #     dx_1_controlled = self.a * (2 * state1[3] * state1[3] * (state1[1] - state1[0]) + self.d * state1[0])+self.u1*10
#     #     dx_2_controlled = self.b * (2 * state1[3] * state1[3] * (state1[0] - state1[1]) - state1[2])
#     #     dx_3_controlled = self.c * (state1[1] - self.h * state1[2])
#     #     dx_4_controlled = state1[1] - state1[0] - 0.01 * state1[3]
#     #     state1[0]=state1[0]+dx_1_controlled*0.001
#     #     state1[1] = state1[1] + dx_2_controlled * 0.001
#     #     state1[2] = state1[2] + dx_3_controlled * 0.001
#     #     state1[3]=state1[3]+dx_4_controlled*0.001
#     #     self.state1=state1
#     #
#     #     dx_1_controlled = self.a * (2 * state1[3] * state1[3] * (state1[1] - state1[0]) + self.d * state1[0])
#     #     dx_2_controlled = self.b * (2 * state1[3] * state1[3] * (state1[0] - state1[1]) - state1[2])
#     #     dx_3_controlled = self.c * (state1[1] - self.h * state1[2])
#     #     dx_4_controlled = state1[1] - state1[0] - 0.01 * state1[3]
#     #     self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled, dx_3_controlled,dx_4_controlled]
#     #     # 更新环境状态
#     #     self.state = self.state0 - self.state2
#     #     now=self._get_observation()
#     #     reward = -sum(abs(x) for x in now[0:4])  # 使用简单的欧氏距离作为奖励
#     #     self.t=self.t+0.001
#     #     # if abs(self.u1-self.input_min)<10 or abs(self.u1-self.input_max)<10:
#     #     #     reward=reward-1
#     #     # if abs(self.u2-self.input_min)<10 or abs(self.u2-self.input_max)<10:
#     #     #     reward=reward-1
#     #     if self.t ==10 or reward<-1e6:
#     #         done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
#     #     else:
#     #         done = False
#     #     return self._get_observation(), reward, done, {}
#     #
#     #
#     # def render(self, mode='human'):
#     #     pass  # 可以添加可视化功能在这里
#
#     # 两个系统的同步
#
#     metadata = {'render.modes': ['human', 'rgb_array']}
#
#     def __init__(self, ):
#
#         self.input_min = -2
#         self.input_max = 2
#         self.state_dim = 8
#         self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
#         self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
#         self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
#         self.state = None
#         self.state0 = None
#         self.state1 = None
#         self.state2 = None
#         self.dis = 0
#         self.u1 = 0
#         self.u2 = 0
#         self.u3 = 0
#         self.t = 0
#         self.a = 30
#         self.b = 1
#         self.c = 36
#         self.d = 0.5
#         self.h = 0.003
#
#     def reset(self):
#         # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
#         # state1=np.random.uniform(low=0, high=5, size=(4,))
#         # state2 = np.random.uniform(low=0, high=5, size=(4,))
#         state1 = np.array([0, 0.01, 0.01, 0])
#         state2 = np.array([2,3.01,4.01,1])
#         self.state1 = state1
#         dx_1_controlled = self.a * (2 * state1[3] * state1[3] * (state1[1] - state1[0]) + self.d * state1[0])
#         dx_2_controlled = self.b * (2 * state1[3] * state1[3] * (state1[0] - state1[1]) - state1[2])
#         dx_3_controlled = self.c * (state1[1] - self.h * state1[2])
#         dx_4_controlled = state1[1] - state1[0] - 0.01 * state1[3]
#         self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled, dx_3_controlled,
#                        dx_4_controlled]
#         self.state2 = state2
#         dx_1_controlled_2 = self.a * (2 * state2[3] * state2[3] * (state2[1] - state2[0]) + self.d * state2[0])
#         dx_2_controlled_2 = self.b * (2 * state2[3] * state2[3] * (state2[0] - state2[1]) - state2[2])
#         dx_3_controlled_2 = self.c * (state2[1] - self.h * state2[2])
#         dx_4_controlled_2 = state2[1] - state2[0] - 0.01 * state2[3]
#         self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2,
#                        dx_3_controlled_2,
#                        dx_4_controlled_2]
#         # 改到这环境从三维改成六维
#
#         self.t = 0
#         return self._get_observation()
#
#     def _get_observation(self):
#         return np.array(self.state0) - np.array(self.state2)
#
#     def get_current(self):
#         return [self.state1[0], self.state2[0]]
#
#     def get_current1(self):
#         return [self.state1[1], self.state2[1]]
#
#     def get_current2(self):
#         return [self.state1[2], self.state2[2]]
#
#     def get_current3(self):
#         return [self.state1[3], self.state2[3]]
#
#     def step(self, action):
#         # 将控制输入应用到受控系统，并模拟动态过程
#         self.u1 = np.clip(action[0], self.input_min, self.input_max)
#         self.u2 = np.clip(action[1], self.input_min, self.input_max)
#         self.u3 = np.clip(action[2], self.input_min, self.input_max)
#         self.state = self._get_observation()
#         # 计算state1
#         state1 = self.state1
#         # self.target_system_noise = np.random.normal(loc=0,scale=3, size=(3,))
#         dx_1_controlled = self.a * (2 * state1[3] * state1[3] * (state1[1] - state1[0]) + self.d * state1[0])
#         dx_2_controlled = self.b * (2 * state1[3] * state1[3] * (state1[0] - state1[1]) - state1[2])
#         dx_3_controlled = self.c * (state1[1] - self.h * state1[2])
#         dx_4_controlled = state1[1] - state1[0] - 0.01 * state1[3]
#         state1[0] = state1[0] + dx_1_controlled * 0.001
#         state1[1] = state1[1] + dx_2_controlled * 0.001
#         state1[2] = state1[2] + dx_3_controlled * 0.001
#         state1[3] = state1[3] + dx_4_controlled * 0.001
#         self.state1 = state1
#
#         dx_1_controlled = self.a * (2 * state1[3] * state1[3] * (state1[1] - state1[0]) + self.d * state1[0])
#         dx_2_controlled = self.b * (2 * state1[3] * state1[3] * (state1[0] - state1[1]) - state1[2])
#         dx_3_controlled = self.c * (state1[1] - self.h * state1[2])
#         dx_4_controlled = state1[1] - state1[0] - 0.01 * state1[3]
#         self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled,
#                        dx_3_controlled,
#                        dx_4_controlled]
#
#         # 计算state2
#         state2 = self.state2
#         self.target_system_noise = np.random.normal(loc=0,scale=0.5, size=(3,))
#         dx_1_controlled_2 = self.a * (2 * state2[3] * state2[3] * (state2[1] - state2[0]) + self.d * state2[0])+ self.u1*100+self.target_system_noise[0]
#         dx_2_controlled_2 = self.b * (2 * state2[3] * state2[3] * (state2[0] - state2[1]) - state2[2]) + self.u2*100+self.target_system_noise[1]
#         dx_3_controlled_2 = self.c * (state2[1] - self.h * state2[2])
#         dx_4_controlled_2 = state2[1] - state2[0] - 0.01 * state2[3]+self.u3*100+self.target_system_noise[2]
#         state2[0] = state2[0] + dx_1_controlled_2 * 0.001
#         state2[1] = state2[1] + dx_2_controlled_2 * 0.001
#         state2[2] = state2[2] + dx_3_controlled_2 * 0.001
#         state2[3] = state2[3] + dx_4_controlled_2 * 0.001
#         self.state2 = state2
#
#         dx_1_controlled_2 = self.a * (2 * state2[3] * state2[3] * (state2[1] - state2[0]) + self.d * state2[0])
#         dx_2_controlled_2 = self.b * (2 * state2[3] * state2[3] * (state2[0] - state2[1]) - state2[2])
#         dx_3_controlled_2 = self.c * (state2[1] - self.h * state2[2])
#         dx_4_controlled_2 = state2[1] - state2[0] - 0.01 * state2[3]
#         self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2, dx_3_controlled_2,
#                        dx_4_controlled_2]
#         # 更新环境状态
#         self.state = np.array(self.state0) -np.array(self.state2)
#         now = self._get_observation()
#         reward = -sum(abs(x) for x in now[0:4])  # 使用简单的欧氏距离作为奖励
#         self.t = self.t + 0.001
#         # if abs(self.u1-self.input_min)<10 or abs(self.u1-self.input_max)<10:
#         #     reward=reward-1
#         # if abs(self.u2-self.input_min)<10 or abs(self.u2-self.input_max)<10:
#         #     reward=reward-1
#         if self.t == 5 or reward < -1e6:
#             done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
#         else:
#             done = False
#         return self._get_observation(), reward, done, {}
#
#     def render(self, mode='human'):
#         pass  # 可以添加可视化功能在这里




class lorenzEnv_transient(gym.Env):

    # 两个系统的同步

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ):

        self.input_min = -2
        self.input_max = 2
        self.state_dim = 8
        self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
        self.state = None
        self.state0 = None
        self.state1 = None
        self.state2 = None
        self.dis = 0
        self.u1 = 0
        self.u2 = 0
        self.u3 = 0
        self.t = 0
        self.a = 10
        self.b = 8/3
        self.c = 28
        self.r = -1

    def reset(self):
        # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
        state1=np.random.uniform(low=0, high=5, size=(4,))
        state2 = np.random.uniform(low=0, high=5, size=(4,))
        self.state1 = state1
        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        dx_2_controlled = self.c*state1[0]-state1[1]-state1[0]*state1[2]
        dx_3_controlled = state1[0]*state1[1]-self.b*state1[2]
        dx_4_controlled = -state1[0]*state1[1]-self.b*state1[2]
        self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled, dx_3_controlled,
                       dx_4_controlled]
        self.state2 = state2
        dx_1_controlled_2 = self.a * (state2[1] - state2[0]) + state2[3]
        dx_2_controlled_2 = self.c * state2[0] - state2[1] - state2[0] * state2[2]
        dx_3_controlled_2 = state2[0] * state2[1] - self.b * state2[2]
        dx_4_controlled_2 = -state2[0] * state2[1] - self.b * state2[2]
        self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2,
                       dx_3_controlled_2,
                       dx_4_controlled_2]
        # 改到这环境从三维改成六维

        self.t = 0
        return self._get_observation()

    def _get_observation(self):
        return np.array(self.state0) - np.array(self.state2)

    def get_current(self):
        return [self.state1[0], self.state2[0]]

    def get_current1(self):
        return [self.state1[1], self.state2[1]]

    def get_current2(self):
        return [self.state1[2], self.state2[2]]

    def get_current3(self):
        return [self.state1[3], self.state2[3]]

    def step(self, action):
        # 将控制输入应用到受控系统，并模拟动态过程
        self.u1 = np.clip(action[0], self.input_min, self.input_max)
        self.u2 = np.clip(action[1], self.input_min, self.input_max)
        self.u3 = np.clip(action[2], self.input_min, self.input_max)
        self.state = self._get_observation()
        # 计算state1
        state1 = self.state1
        # self.target_system_noise = np.random.normal(loc=0,scale=3, size=(3,))
        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        dx_2_controlled = self.c * state1[0] - state1[1] - state1[0] * state1[2]
        dx_3_controlled = state1[0] * state1[1] - self.b * state1[2]
        dx_4_controlled = -state1[0] * state1[1] - self.b * state1[2]
        state1[0] = state1[0] + dx_1_controlled * 0.001
        state1[1] = state1[1] + dx_2_controlled * 0.001
        state1[2] = state1[2] + dx_3_controlled * 0.001
        state1[3] = state1[3] + dx_4_controlled * 0.001
        self.state1 = state1

        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        dx_2_controlled = self.c * state1[0] - state1[1] - state1[0] * state1[2]
        dx_3_controlled = state1[0] * state1[1] - self.b * state1[2]
        dx_4_controlled = -state1[0] * state1[1] - self.b * state1[2]
        self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled,
                       dx_3_controlled,
                       dx_4_controlled]

        # 计算state2
        state2 = self.state2
        self.target_system_noise = np.random.normal(loc=0,scale=0.5, size=(3,))
        dx_1_controlled_2 = self.a * (state2[1] - state2[0]) + state2[3]
        dx_2_controlled_2 = self.c * state2[0] - state2[1] - state2[0] * state2[2]
        dx_3_controlled_2 = state2[0] * state2[1] - self.b * state2[2]
        dx_4_controlled_2 = -state2[0] * state2[1] - self.b * state2[2]
        state2[0] = state2[0] + dx_1_controlled_2 * 0.001
        state2[1] = state2[1] + dx_2_controlled_2 * 0.001
        state2[2] = state2[2] + dx_3_controlled_2 * 0.001
        state2[3] = state2[3] + dx_4_controlled_2 * 0.001
        self.state2 = state2

        dx_1_controlled_2 = self.a * (state2[1] - state2[0]) + state2[3]
        dx_2_controlled_2 = self.c * state2[0] - state2[1] - state2[0] * state2[2]
        dx_3_controlled_2 = state2[0] * state2[1] - self.b * state2[2]
        dx_4_controlled_2 = -state2[0] * state2[1] - self.b * state2[2]
        self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2, dx_3_controlled_2,
                       dx_4_controlled_2]
        # 更新环境状态
        self.state = np.array(self.state0) -np.array(self.state2)
        now = self._get_observation()
        reward = -sum(abs(x) for x in now[0:4])  # 使用简单的欧氏距离作为奖励
        self.t = self.t + 0.001
        # if abs(self.u1-self.input_min)<10 or abs(self.u1-self.input_max)<10:
        #     reward=reward-1
        # if abs(self.u2-self.input_min)<10 or abs(self.u2-self.input_max)<10:
        #     reward=reward-1
        if self.t == 5 or reward < -1e6:
            done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
        else:
            done = False
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass  # 可以添加可视化功能在这里



