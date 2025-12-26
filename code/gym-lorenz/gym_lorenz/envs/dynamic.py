import gym

import numpy as np

class lorenzEnv_transient(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4, xs=-1.6, input_range=[-10.0, 10.0], id_range=None,
                 noise_std=0.22, gamma=0.9):
        self.a, self.b, self.c, self.d, self.r, self.s, self.xs = a, b, c, d, r, s, xs
        self.input_min = -500.0
        self.input_max = 500.0
        self.id_range = id_range if id_range is not None else [0, 5]
        self.noise_std = noise_std
        self.gamma = gamma
        self.state_dim = 6  # 包含两个HR系统状态差分量
        self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
        self.state = None#状态变量六维
        self.state0=None#六维
        self.state1 = None#三维
        self.state2 = None#六维
        self.input_signal = 3.2
        self.input_control = 0
        self.u1 = 0
        self.u2 = 0
        self.u3 = 0
        self.t=0
        self.u = 10
        self.i = 28
        self.o = 8 / 3

    def reset(self):
        # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
        state1=np.random.uniform(low=-30, high=30, size=(3,))
        self.state1=state1
        dxdt_controlled=self.u * (state1[1] - state1[0])
        dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
        dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
        self.state0 = [state1[0],state1[1],state1[2],dxdt_controlled,dydt_controlled,dzdt_controlled]
        #改到这环境从三维改成六维
        self.state2 = np.array([0, 0, 0, 0, 0, 0])
        self.target_system_noise = np.random.normal(scale=self.noise_std, size=(3,))
        self.t = 0
        return self._get_observation()

    def _get_observation(self):
        return self.state0 - self.state2

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
        self.u3 = np.clip(action[2], self.input_min, self.input_max)
        self.state = self._get_observation()
        #print(self.state1)
        # 计算奖励函数
        state1 = self.state1
        dxdt_controlled = self.u * (state1[1] - state1[0])
        dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
        dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
        state1[0]=state1[0]+dxdt_controlled*0.01+self.u1
        state1[1] = state1[1] + dydt_controlled * 0.01 + self.u2
        state1[2] = state1[2] + dzdt_controlled * 0.01 + self.u3
        self.state1=state1
        dxdt_controlled = self.u * (state1[1] - state1[0])
        dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
        dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
        self.state0 = [state1[0],state1[1],state1[2],dxdt_controlled, dydt_controlled, dzdt_controlled]
        # 更新环境状态
        self.state = self.state0 - self.state2
        now=self._get_observation()
        reward = -sum(abs(x) for x in now[0:3])  # 使用简单的欧氏距离作为奖励
        self.t=self.t+0.01
        if self.t ==10 :
            done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
        else:
            done = False
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass  # 可以添加可视化功能在这里


    #两个lorenz同步
    # import python libraries
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

        def __init__(self, id_range=None,
                     noise_std=0.22, gamma=0.9):

            self.input_min = -500.0
            self.input_max = 500.0
            self.id_range = id_range if id_range is not None else [0, 5]
            self.noise_std = noise_std
            self.gamma = gamma
            self.state_dim = 6  # 包含两个HR系统状态差分量
            self.action_dim = (self.input_max - self.input_min)  # 控制输入范围
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
            # self.action_space = gym.spaces.Box(self.input_min, self.input_max, shape=(3,), dtype=np.float32)
            self.state = None  # 状态变量六维
            self.state0 = None  # 六维
            self.state1 = None  # 三维
            self.state12 = None  # 三维
            self.state2 = None  # 六维
            self.u1 = 0
            self.u2 = 0
            self.u3 = 0
            self.t = 0
            self.u = 10
            self.i = 28
            self.o = 8 / 3
            self.w = 1.0
            self.af = 0.165
            self.bt = 0.2
            self.gm = 10

        def reset(self):
            # 初始化两个HR系统的状态并计算初始误差向量作为环境状态
            state1 = np.random.uniform(low=-20, high=20, size=(3,))
            self.state1 = state1
            dxdt_controlled = self.u * (state1[1] - state1[0])
            dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
            dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
            self.state0 = [state1[0], state1[1], state1[2], dxdt_controlled, dydt_controlled, dzdt_controlled]
            # 改到这环境从三维改成六维
            state12 = np.random.uniform(low=-20, high=20, size=(3,))
            self.state12 = state12
            dxdt_controlled1 = self.u * (state12[1] - state12[0])
            dydt_controlled1 = self.i * state12[0] - state12[1] - state12[0] * state12[2]
            dzdt_controlled1 = state12[0] * state12[1] - self.o * state12[2]
            self.state2 = [state12[0], state12[1], state12[2], dxdt_controlled1, dydt_controlled1, dzdt_controlled1]
            self.t = 0
            return self._get_observation()

        def _get_observation(self):
            state0 = np.array(self.state0)
            state2 = np.array(self.state2)
            return state0 - state2

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
            self.u3 = np.clip(action[2], self.input_min, self.input_max)
            self.state = self._get_observation()
            # 计算奖励函数
            state1 = self.state1
            dxdt_controlled = self.u * (state1[1] - state1[0])
            dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
            dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
            state1[0] = state1[0] + dxdt_controlled * 0.01 + self.u1
            state1[1] = state1[1] + dydt_controlled * 0.01 + self.u2
            state1[2] = state1[2] + dzdt_controlled * 0.01 + self.u3
            self.state1 = state1
            dxdt_controlled = self.u * (state1[1] - state1[0])
            dydt_controlled = self.i * state1[0] - state1[1] - state1[0] * state1[2]
            dzdt_controlled = state1[0] * state1[1] - self.o * state1[2]
            self.state0 = [state1[0], state1[1], state1[2], dxdt_controlled, dydt_controlled, dzdt_controlled]

            # state12 = self.state12
            # dxdt_controlled1 = -self.w * state12[1] - state12[2]
            # dydt_controlled1 = self.w * state12[0] + self.af * state12[1]
            # dzdt_controlled1 = self.bt + state12[2] * (state12[0] - self.gm)
            # state12[0] = state12[0] + dxdt_controlled1 * 0.01
            # state12[1] = state12[1] + dydt_controlled1 * 0.01
            # state12[2] = state12[2] + dzdt_controlled1 * 0.01
            # self.state12 = state12
            # dxdt_controlled1 = -self.w * state12[1] - state12[2]
            # dydt_controlled1 = self.w * state12[0] + self.af * state12[1]
            # dzdt_controlled1 = self.bt + state12[2] * (state12[0] - self.gm)
            # self.state2 = [state12[0], state12[1], state12[2], dxdt_controlled1, dydt_controlled1, dzdt_controlled1]

            # lorenz另外一个
            # state12 = self.state12
            # dxdt_controlled1 = self.u * (state12[1] - state12[0])
            # dydt_controlled1 = self.i * state12[0] - state12[1] - state12[0] * state12[2]
            # dzdt_controlled1 = state12[0] * state12[1] - self.o * state12[2]
            # state12[0] = state12[0] + dxdt_controlled1 * 0.01
            # state12[1] = state12[1] + dydt_controlled1 * 0.01
            # state12[2] = state12[2] + dzdt_controlled1 * 0.01
            # self.state12 = state12
            # dxdt_controlled1 = self.u * (state12[1] - state12[0])
            # dydt_controlled1 = self.i * state12[0] - state12[1] - state12[0] * state12[2]
            # dzdt_controlled1 = state12[0] * state12[1] - self.o * state12[2]
            # self.state2 = [state12[0], state12[1], state12[2], dxdt_controlled1, dydt_controlled1, dzdt_controlled1]

            # 更新环境状态
            self.state = np.array(self.state0) - np.array(self.state2)
            now = self._get_observation()
            reward = -sum(abs(x) for x in now[0:3])  # 使用简单的欧氏距离作为奖励
            self.t = self.t + 0.01
            if self.t == 10:
                done = True  # 这里可以根据实际情况设置完成条件，例如达到一定同步程度时结束episode
            else:
                done = False
            return self._get_observation(), reward, done, {}

        def render(self, mode='human'):
            pass  # 可以添加可视化功能在这里




