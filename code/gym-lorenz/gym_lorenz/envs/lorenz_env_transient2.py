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


# 注意：这里类名保持为您原本的 lorenzEnv_transient
class lorenzEnv_transient(gym.Env):
    # 两个系统的同步 (Hyper-Lorenz System)

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

        # --- 修改部分：超混沌 Lorenz 系统参数 (Jia 2007) ---
        self.a = 10.0
        self.b = 8.0 / 3.0  # 约等于 2.666
        self.c = 28.0
        self.k = 1.3  # 新增参数 k (原系统的 d 和 h 参数不再使用)
        # -----------------------------------------------

    def reset(self):
        # 初始化两个系统的状态并计算初始误差向量作为环境状态
        state1 = np.random.uniform(low=0, high=5, size=(4,))  # 稍微调整了范围以适应Lorenz
        state2 = np.random.uniform(low=0, high=5, size=(4,))

        self.state1 = state1

        # --- 修改部分：Master System 方程 (State 1) ---
        # 对应变量: state1[0]=x, state1[1]=y, state1[2]=z, state1[3]=w
        # dx = a(y - x) + w
        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        # dy = cx - y - xz
        dx_2_controlled = self.c * state1[0] - state1[1] - state1[0] * state1[2]
        # dz = xy - bz
        dx_3_controlled = state1[0] * state1[1] - self.b * state1[2]
        # dw = -kw + xz
        dx_4_controlled = -self.k * state1[3] + state1[0] * state1[2]
        # -------------------------------------------

        self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled, dx_3_controlled,
                       dx_4_controlled]
        self.state2 = state2

        # --- 修改部分：Slave System 方程 (State 2) ---
        dx_1_controlled_2 = self.a * (state2[1] - state2[0]) + state2[3]
        dx_2_controlled_2 = self.c * state2[0] - state2[1] - state2[0] * state2[2]
        dx_3_controlled_2 = state2[0] * state2[1] - self.b * state2[2]
        dx_4_controlled_2 = -self.k * state2[3] + state2[0] * state2[2]
        # -------------------------------------------

        self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2,
                       dx_3_controlled_2,
                       dx_4_controlled_2]

        self.t = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.array(self.state0) - np.array(self.state2)
        # === 添加：数值保护 ===
        obs = np.clip(obs, -1e6, 1e6)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs.astype(np.float32)

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

        # --- 计算 state1 (Master System) ---
        state1 = self.state1
        # 计算导数
        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        dx_2_controlled = self.c * state1[0] - state1[1] - state1[0] * state1[2]
        dx_3_controlled = state1[0] * state1[1] - self.b * state1[2]
        dx_4_controlled = -self.k * state1[3] + state1[0] * state1[2]

        # Euler 积分更新状态
        state1[0] = state1[0] + dx_1_controlled * 0.001
        state1[1] = state1[1] + dx_2_controlled * 0.001
        state1[2] = state1[2] + dx_3_controlled * 0.001
        state1[3] = state1[3] + dx_4_controlled * 0.001
        # === 添加：裁剪 state1 防止发散 ===
        state1 = np.clip(state1, -1000, 1000)
        self.state1 = state1

        # 更新观测用的导数 (用于 state0)
        dx_1_controlled = self.a * (state1[1] - state1[0]) + state1[3]
        dx_2_controlled = self.c * state1[0] - state1[1] - state1[0] * state1[2]
        dx_3_controlled = state1[0] * state1[1] - self.b * state1[2]
        dx_4_controlled = -self.k * state1[3] + state1[0] * state1[2]

        self.state0 = [state1[0], state1[1], state1[2], state1[3], dx_1_controlled, dx_2_controlled,
                       dx_3_controlled,
                       dx_4_controlled]

        # --- 计算 state2 (Slave System with Control) ---
        state2 = self.state2
        self.target_system_noise = np.random.normal(loc=0, scale=0.5, size=(4,))

        # 计算带控制的导数 (Hyper-Lorenz + Control)
        # u1 加在 dx (state2[0]) 上
        # u2 加在 dy (state2[1]) 上
        # u3 加在 dw (state2[3]) 上
        # mu = 120
        mu = 40
        dx_1_controlled_2 = (self.a * (state2[1] - state2[0]) + state2[3]) + self.u1 * mu
        dx_2_controlled_2 = (self.c * state2[0] - state2[1] - state2[0] * state2[2]) + self.u2 * mu
        dx_3_controlled_2 = (state2[0] * state2[1] - self.b * state2[2]) # + self.u2 * 120
        dx_4_controlled_2 = (-self.k * state2[3] + state2[0] * state2[2]) + self.u3 * mu
        # Euler 积分更新状态
        state2[0] = state2[0] + dx_1_controlled_2 * 0.001
        state2[1] = state2[1] + dx_2_controlled_2 * 0.001
        state2[2] = state2[2] + dx_3_controlled_2 * 0.001
        state2[3] = state2[3] + dx_4_controlled_2 * 0.001
        # === 添加：裁剪 state2 前4维防止发散 ===
        state2[:4] = np.clip(state2[:4], -1000, 1000)
        self.state2 = state2

        # 更新观测用的导数 (不含控制输入，与 reset 逻辑一致)
        dx_1_controlled_2 = self.a * (state2[1] - state2[0]) + state2[3]
        dx_2_controlled_2 = self.c * state2[0] - state2[1] - state2[0] * state2[2]
        dx_3_controlled_2 = state2[0] * state2[1] - self.b * state2[2]
        dx_4_controlled_2 = -self.k * state2[3] + state2[0] * state2[2]

        self.state2 = [state2[0], state2[1], state2[2], state2[3], dx_1_controlled_2, dx_2_controlled_2,
                       dx_3_controlled_2,
                       dx_4_controlled_2]

        # 更新环境状态
        self.state = np.array(self.state0) - np.array(self.state2)
        now = self._get_observation()
        reward = -sum(abs(x) for x in now[0:4]) - sum(abs(x) for x in now[0:4]) ** (1 / 2)
        self.t = self.t + 0.001

        # === 修改：使用 >= 代替 == 进行浮点数比较 ===
        if self.t == 2 or reward < -1e3:
            done = True
        else:
            done = False
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        pass
    