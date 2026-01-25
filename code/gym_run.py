# import the necessary libraries, the environment, and RL policy
import math
import os
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from mpl_toolkits.mplot3d import Axes3D
import gym
import csv
import gym_lorenz
import numpy

import matplotlib
# 必须在 import pyplot 之前设置 backend，'Agg' 是用于生成图像文件的非交互后端
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import pandas as pd

import torch as th
import torch.cuda
import networkx as nx
from stable_baselines3.common.callbacks import BaseCallback


# 添加自定义回调类 - 不依赖Monitor
class RewardCallback(BaseCallback):
    """
    自定义回调函数，用于记录训练过程中的奖励
    """

    def __init__(self, check_freq=1000, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # 累积当前步的奖励
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # 检查episode是否结束
        if self.locals['dones'][0]:
            # 记录episode信息
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(
                    f"Episode {len(self.episode_rewards)}: reward={self.current_episode_reward:.2f}, length={self.current_episode_length}")

            # 重置计数器
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def get_data(self):
        """返回收集的数据"""
        return self.timesteps, self.episode_rewards, self.episode_lengths


def plot_rewards(timesteps, rewards, save_path=None):
    """
    绘制训练奖励曲线

    参数:
    - timesteps: 时间步列表
    - rewards: 奖励列表
    - save_path: 保存路径，None则显示
    """
    # 设置字体
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.style'] = 'normal'

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 绘制原始奖励
    ax.plot(timesteps, rewards, alpha=0.3, color='blue', label='Episode Reward')

    # 计算并绘制移动平均
    if len(rewards) > 10:
        window_size = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        moving_avg_steps = timesteps[window_size - 1:]
        ax.plot(moving_avg_steps, moving_avg, linewidth=2, color='red',
                label=f'Moving Average (window={window_size})')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奖励曲线已保存到: {save_path}")
    else:
        plt.show()


class CustomAttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with self-attention mechanism.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super(CustomAttentionFeaturesExtractor, self).__init__(observation_space, features_dim)
        # 定义前馈网络部分
        self.fc1 = torch.nn.Linear(observation_space.shape[0], 128)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(128, features_dim)  # 隐藏层到特征层的前半部分
        # 定义自注意力层
        self.attention_layer = torch.nn.MultiheadAttention(features_dim, num_heads=4, batch_first=True)
        # 注意力层后的处理，可以选择再次通过一层网络或者直接使用
        self.post_attention_fc = torch.nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: (batch_size, obs_dim)
        :return: (batch_size, features_dim)
        """
        x = torch.relu(self.fc1(observations))  # 第一层全连接
        x = torch.relu(self.fc2(x))  # 第二层全连接，准备进入注意力层
        # 重塑张量以适应自注意力层的输入格式
        x = x.unsqueeze(1)  # 添加序列维度 (batch_size, seq_len=1, features_dim)
        x, _ = self.attention_layer(x, x, x)  # 自注意力层
        x = x.squeeze(1)  # 移除序列维度
        x = torch.relu(self.post_attention_fc(x))  # 可选的后处理层
        return x


def function(save_name):

    env = gym.make('lorenz_transient-v0')
    env = DummyVecEnv([lambda: env])

    reward_callback = RewardCallback(verbose=1)
    n_actions = env.action_space.shape[-1]

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./lorenztensorboard2/",policy_kwargs=policy_kwargs,
                learning_rate=5e-5)

    print("开始训练...")
    # 开始训练，传入回调
    model.learn(total_timesteps=1000000, callback=reward_callback)
    # 保存模型
    model.save(save_name)
    print("模型已保存")

    # 获取并绘制奖励曲线
    timesteps, rewards, lengths = reward_callback.get_data()
    print(f"\n训练统计:")
    print(f"- 总训练步数: {timesteps[-1] if timesteps else 0}")
    print(f"- 完成的 episodes: {len(rewards)}")

    if len(rewards) > 0:
        print(f"- 平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"- 最大奖励: {np.max(rewards):.2f}")
        print(f"- 最小奖励: {np.min(rewards):.2f}")
        if len(rewards) >= 100:
            print(f"- 最后100个episodes平均奖励: {np.mean(rewards[-100:]):.2f}")

        # 绘制并保存奖励曲线
        plot_rewards(timesteps, rewards, save_path=f'training_plot/{save_name}_reward_curve.png')
        print(f"奖励曲线已保存为: {save_name}_reward_curve.png")
    else:
        print("警告：未收集到奖励数据，请检查环境配置")

    return reward_callback


def testPPO1(model_name0):
    # 1. 加载环境和模型
    env = gym.make('lorenz_transient-v0')  # 确保环境ID正确
    model = PPO.load(model_name0, env, verbose=1)

    # 简单评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward / 1000:.2f} ± {std_reward / 1000:.2f}")

    # 2. 初始化数据存储列表 (扩充为 4个状态 + 3个控制)
    all_list_obs1 = [[] for _ in range(10)]  # x error
    all_list_obs2 = [[] for _ in range(10)]  # y error
    all_list_obs3 = [[] for _ in range(10)]  # z error
    all_list_obs4 = [[] for _ in range(10)]  # w error (新增)

    all_list_act1 = [[] for _ in range(10)]  # u1
    all_list_act2 = [[] for _ in range(10)]  # u2
    all_list_act3 = [[] for _ in range(10)]  # u3 (新增)

    list_inital = []  # 用于存储每个Episode的初始状态用于图例

    # 3. 开始 10 次测试循环
    for j in range(10):
        obs = env.reset()

        list_obs1, list_obs2, list_obs3, list_obs4 = [], [], [], []
        list_act1, list_act2, list_act3 = [], [], []

        for i in range(2000):  # 运行 200 步
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

            # 记录 4 个维度的误差
            list_obs1.append(obs[0])  # x
            list_obs2.append(obs[1])  # y
            list_obs3.append(obs[2])  # z
            list_obs4.append(obs[3])  # w (新增)

            # 记录 3 个维度的控制力


            # 记录初始值 (只在第一步记录)
            if i == 0:
                # 存储结构: [x0, y0, z0, w0, u1_0, u2_0, u3_0]
                list_inital.append([obs[0], obs[1], obs[2], obs[3], action[0], action[1], action[2]])


            # 将单次测试的数据存入总表
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_obs4[j] = list_obs4

        all_list_act1[j] = list_act1
        all_list_act2[j] = list_act2
        all_list_act3[j] = list_act3

    # 4. 绘图设置
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.style'] = 'normal'

    # 5. 绘图循环：现在需要画 7 张图 (0-3是误差, 4-6是控制力)
    # k=0: x, k=1: y, k=2: z, k=3: w
    # k=4: u1, k=5: u2, k=6: u3
    for k in range(4):
        fig, axs = plt.subplots(1, 1, figsize=(15, 6))

        for j in range(10):  # 遍历 10 次测试
            # 数据选择逻辑
            if k == 0:
                data = all_list_obs1[j]
            elif k == 1:
                data = all_list_obs2[j]
            elif k == 2:
                data = all_list_obs3[j]
            elif k == 3:
                data = all_list_obs4[j]  # 新增 w

            # 图例文字与数值缩放逻辑
            # k <= 3 代表是状态误差 (x, y, z, w)
            if k <= 3:
                text = "Error"
                # 对应 list_inital 的索引 0,1,2,3
                num = round(list_inital[j][k], 1)

            if k == 2:
                # 如果是 Z 轴 (k=2)，使用全部 2000 步数据
                plot_data =data
                steps = 2000
            else:
                # 如果是 X, Y, W (k=0,1,3)，只切片取前 200 步
                plot_data = data[:200]
                steps = 200



            time_axis = [(t / 1000) for t in range(0, len(plot_data))]
            axs.plot(time_axis, plot_data, label=f"Test {j + 1}(inital {text}={num})")

        axs.legend(prop={'size': 15}, loc='upper right')

        # 设置标签 Titles and Labels
        if k == 0:
            axs.set_ylabel(f"Error x")
        elif k == 1:
            axs.set_ylabel(f"Error y")
        elif k == 2:
            axs.set_ylabel(f"Error z")
        elif k == 3:
            axs.set_ylabel(f"Error w")  # 新增
        elif k == 4:
            axs.set_ylabel(f"Control u1")  # 修正了之前的 Label 错误
        elif k == 5:
            axs.set_ylabel(f"Control u2")
        elif k == 6:
            axs.set_ylabel(f"Control u3")  # 新增


        axs.set_xlabel("Time(s)")


        plt.show()  # 显示图片
        _pend = "_mu40"
        plt.savefig(f"training_plot/{model_name0}{_pend}_err.png") 
        print(f"图像已保存为 {model_name0}{_pend}_err.png")
        # 关闭图表以释放内存
        plt.close()
        
        save_dir = "export_errors_csv"
        os.makedirs(save_dir, exist_ok=True)

        # 将误差数组转成矩阵结构，保证长度一致
        min_len = min(len(traj) for traj in all_list_obs1)  # 最短序列
        err_x = np.array([traj[:min_len] for traj in all_list_obs1])
        err_y = np.array([traj[:min_len] for traj in all_list_obs2])
        err_z = np.array([traj[:min_len] for traj in all_list_obs3])
        err_w = np.array([traj[:min_len] for traj in all_list_obs4])

        # 定义一个保存函数
        def save_csv(filename, data):
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["episode"] + [f"t{i}" for i in range(data.shape[1])]
                writer.writerow(header)

                for ep in range(data.shape[0]):
                    writer.writerow([ep] + list(data[ep]))
            print(f"Saved → {filepath}")

        # 保存 4 组误差 CSV
        save_csv("error_x.csv", err_x)
        save_csv("error_y.csv", err_y)
        save_csv("error_z.csv", err_z)
        save_csv("error_w.csv", err_w)

        print("All CSV export completed.")


def testPPO3(model_name0):
    env = gym.make('lorenz_transient-v0')
    model = A2C.load(model_name0, env, verbose=1)
    print(model.policy)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward/1000:.2f} ± {std_reward/1000:.2f}")

    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(10)]
    all_list_obs2 = [[] for _ in range(10)]
    all_list_obs3 = [[] for _ in range(10)]
    all_list_act1 = [[] for _ in range(10)]
    all_list_act2 = [[] for _ in range(10)]

    list_inital=[]
    # list_inital = [[14.3, 19.3, -29.9], [-17.7, -0.1, 15.2]
    #     , [29.4, 28.6, 7.1], [19.3, 24.1, -10.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 29.5], [12.6, 27.7, -10.8],
    #                [11.6, -4.8, 18.9], [13.5, 26.9, -5.9]]

    for j in range(10):
        obs = env.reset()
        print(obs)
        # dxdt_controlled = -list_inital[j][0] + list_inital[j][1] * list_inital[j][2]
        # dydt_controlled = -list_inital[j][1] - list_inital[j][0] * list_inital[j][2] + 20 * list_inital[j][2]
        # dzdt_controlled = 5.46 * (list_inital[j][1] - list_inital[j][2])
        # obs = ([list_inital[j][0], list_inital[j][1], list_inital[j][2], dxdt_controlled, dydt_controlled, dzdt_controlled])
        # print(obs)
        list_obs1, list_obs2, list_obs3 = [], [], []
        list_act1= []



        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_act1.append(action[0])
            if i==0:
                list_inital.append([obs[0],obs[1],obs[2],action[0]])


        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_act1[j] = list_act1


    # 设置全局字体大小
    plt.rcParams['font.size'] = 18  # 设置默认字体大小为14
    figsize_width = 24
    figsize_height = 6
    # 指定字体文件的完整路径
    font_path = '/usr/share/fonts/truetype/times/times.ttf'
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/times/times.ttf')

    # 创建一个FontProperties对象，指向你的字体文件
    prop = FontProperties(fname=font_path)

    # 更新rcParams以使用指定的字体
    plt.rcParams['font.family'] = prop.get_name()
    print(prop.get_name())

    # plt.rcParams['font.family'] = 'Times New Roman'#(windows下使用)
    # plt.rcParams['font.style'] = 'normal'
    # 分别为每种观测值绘制10条线
    for k in range(4):  # 对应obs[0], obs[1], obs[2]
        fig, axs = plt.subplots(1, 1, figsize=(15, 6))  # 每次只创建一张图
        for j in range(10):
            if k == 0:
                data = all_list_obs1[j]
            elif k == 1:
                data = all_list_obs2[j]
            elif k == 2:
                data = all_list_obs3[j]
            elif k == 3:
                data = all_list_act1[j]
            # if k<=2:
            #     text="Error"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], data,
            #              label=f"Test {j + 1}(inital {text}={round(list_inital[j][k], 1)})")
            # else:
            #     for i in range(len(data)):
            #         test_data.append(data[i]*20)
            #     text="Control Force"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], test_data, label=f"Test {j + 1}(inital {text}={round(list_inital[j][k]*20,1)})")
            if k<=2:
                text="Error"
                num=round(list_inital[j][k],1)
            else:
                num=round(list_inital[j][k]*50,1)
                text="Control Force"
            axs.plot([(i / 100) for i in range(0, len(data))], data, label=f"Test {j + 1}(inital {text}={num})")
            axs.legend(prop={'size': 15}, loc='upper right')  # 'upper right'位置参数使图例显示在右上角
        if k==0:
            #axs.set_title(r"Error in  ")
            axs.set_ylabel(f"x")
        elif k==1:
            #axs.set_title(r"Error in  ")
            axs.set_ylabel(f"y")
        elif k == 2:
            #axs.set_title(r"Error in  ")
            axs.set_ylabel(f"z")
        elif k==3:
            #axs.set_title(f"Control Force of x")
            axs.set_ylabel(f"Error")
        axs.set_xlabel("Time(s)")
        # if k<=2:
        #     axs.set_ylabel(f"Error")
        # else:
        #     axs.set_ylabel(f"control term")

        # # 调用函数
        # write_data_to_file(all_list_act1, all_list_act2, '/tmp/pycharm_project_481/code/dataofresults/ddpg.txt')

        # 将每个列表转换为DataFrame
        df1 = pd.DataFrame(all_list_obs1)
        df2 = pd.DataFrame(all_list_obs2)
        df3 = pd.DataFrame(all_list_obs3)

        # 写入同一个Excel文件的不同sheet中
        # with pd.ExcelWriter('/tmp/pycharm_project_60/code/result.xlsx', engine='openpyxl') as writer:
        #     df1.to_excel(writer, sheet_name='Sheet1', index=False)
        #     df2.to_excel(writer, sheet_name='Sheet2', index=False)
        #     df3.to_excel(writer, sheet_name='Sheet3', index=False)

        # 移除边距
        plt.margins(0, 0)

        # 调整布局，让图表占满整个图片
        plt.tight_layout()

        # 显示当前图
        plt.show()
        plt.savefig("training_plot/pt2.png") 
        print("图像已保存为 pt2.png")
        # 关闭图表以释放内存
        plt.close()



def write_to_excel(filepath, x, sheetName, num):
    # 生成 time 列，从 0 开始，每行增加 0.001
    time = [i * 0.01 for i in range(len(x))]

    if os.path.exists(filepath):
        try:
            # 读取现有的Excel文件
            existing_df = pd.read_excel(filepath, sheet_name=sheetName)
            new_col = pd.DataFrame({num: x})
            # 检查新列长度是否与现有DataFrame匹配
            if len(existing_df) != len(x):
                raise ValueError("New data length does not match existing data length.")
            # 添加新列到现有DataFrame
            existing_df = pd.concat([existing_df, new_col], axis=1)
            df = existing_df
        except Exception as e:
            print(f"Error reading existing Excel file: {e}. Creating new sheet.")
            # 如果读取文件失败，则创建新的DataFrame
            if num == 2:
                df = pd.DataFrame({
                    "time": time,
                    num: x
                })
            else:
                df = pd.DataFrame({
                    num: x
                })
    else:
        # 文件不存在，直接创建新的DataFrame
        if num == 2:
            df = pd.DataFrame({
                "time": time,
                num: x
            })
        else:
            df = pd.DataFrame({
                num: x
            })

    # 写入Excel文件
    with pd.ExcelWriter(filepath, mode='w', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheetName, index=False)


def write_to_excel_2(filepath, x, sheetName, num):
    # 生成 time 列，从 0 开始，每行增加 0.001
    time = [i * 0.001 for i in range(len(x))]

    if os.path.exists(filepath):
        try:
            # 读取现有的Excel文件
            existing_df = pd.read_excel(filepath, sheet_name=sheetName)
            new_col = pd.DataFrame({num: x})
            # 检查新列长度是否与现有DataFrame匹配
            if len(existing_df) != len(x):
                raise ValueError("New data length does not match existing data length.")
            # 添加新列到现有DataFrame
            existing_df = pd.concat([existing_df, new_col], axis=1)
            df = existing_df
        except Exception as e:
            print(f"Error reading existing Excel file: {e}. Creating new sheet.")
            # 如果读取文件失败，则创建新的DataFrame
            if num == 2:
                df = pd.DataFrame({
                    "time": time,
                    num: x
                })
            else:
                df = pd.DataFrame({
                    num: x
                })
    else:
        # 文件不存在，直接创建新的DataFrame
        if num == 2:
            df = pd.DataFrame({
                "time": time,
                num: x
            })
        else:
            df = pd.DataFrame({
                num: x
            })

    # 写入Excel文件
    with pd.ExcelWriter(filepath, mode='w', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheetName, index=False)


def  testyz(model_name0):
    env = gym.make('lorenz_transient-v0')
    print("Environment action space:", env.action_space)
    model = PPO.load(model_name0, env, verbose=1)
    print("Model action space:", model.action_space)
    print(model.policy)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward/1000:.2f} ± {std_reward/1000:.2f}")

    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(10)]
    all_list_obs2 = [[] for _ in range(10)]
    all_list_obs3 = [[] for _ in range(10)]
    all_list_obs4 = [[] for _ in range(10)]
    all_list_act1 = [[] for _ in range(10)]

    list_inital = []

    list_inital = [[ 1.90769,-2.49926,4.04782,2.69377,0.08134,1.67544	,3.33864,1.67153,2.16228,-0.94568],
                   [1.91497,	-2.45559,	1.55886	,-1.3815,	-1.15842	,-4.01014	,4.12956,	0.29897	,0.6683,	-2.08995]
        , [ -0.14655	,-0.96806	,-1.5305,	-3.46583	,3.97548	,0.84039,	1.13859,	3.47758	,2.26294	,-1.23365],
                   [1.39558,	3.5584,	-2.13597,	3.28933,	2.98156,	1.37524,	-1.89389,	1.572,	2.34784,	-0.38377]]

    print(len(list_inital))
    for j in range(10):
        obs = env.reset()
        print(obs)
        dx_1_controlled = 30 * (2 * list_inital[3][j] * list_inital[3][j] * (list_inital[1][j] - list_inital[0][j]) + 0.5 * list_inital[0][j])
        dx_2_controlled = 1 * (2 * list_inital[3][j] * list_inital[3][j] * (list_inital[0][j] - list_inital[1][j]) - list_inital[2][j])
        dx_3_controlled = 36 * (list_inital[1][j] - 0.003 * list_inital[2][j])
        dx_4_controlled = list_inital[1][j] - list_inital[0][j] - 0.01 * list_inital[3][j]
        state0 = [list_inital[0][j], list_inital[1][j], list_inital[2][j], list_inital[3][j], dx_1_controlled, dx_2_controlled, dx_3_controlled,
                       dx_4_controlled]
        dx_1_controlled_2 = 30 * (2 * list_inital[3][j] * list_inital[3][j] * (list_inital[1][j] - list_inital[0][j]) + 0.5 * list_inital[0][j])
        dx_2_controlled_2 = 1 * (2 * list_inital[3][j] * list_inital[3][j] * (list_inital[0][j] - list_inital[1][j]) - list_inital[2][j])
        dx_3_controlled_2 = 36 * (list_inital[1][j] - 0.003 * list_inital[2][j])
        dx_4_controlled_2 = list_inital[1][j] - list_inital[0][j] - 0.01 * list_inital[3][j]
        state2 = [list_inital[0][j], list_inital[1][j], list_inital[2][j], list_inital[3][j], dx_1_controlled_2, dx_2_controlled_2,
                       dx_3_controlled_2,
                       dx_4_controlled_2]
        # dxdt_controlled = -list_inital[j][0] + list_inital[j][1] * list_inital[j][2]
        # dydt_controlled = -list_inital[j][1] - list_inital[j][0] * list_inital[j][2] + 20 * list_inital[j][2]
        # dzdt_controlled = 5.46 * (list_inital[j][1] - list_inital[j][2])
        # obs = ([list_inital[j][0], list_inital[j][1], list_inital[j][2], dxdt_controlled, dydt_controlled, dzdt_controlled])
        # list_inital.append(obs[0:4])
        obs = ([list_inital[0][j], list_inital[1][j], list_inital[2][j], list_inital[3][j],dx_1_controlled-dx_1_controlled_2, dx_2_controlled-dx_2_controlled_2, dx_3_controlled-dx_3_controlled_2,dx_4_controlled-dx_4_controlled_2])
        list_inital.append(obs[0:4])
        list_obs1, list_obs2, list_obs3,list_obs4 = [], [], [],[]
        list_act1,list_act2,list_act3,list_act4= [],[],[],[]

        for i in range(30):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_obs4.append(obs[3])
            list_act1.append(action[0])
            if i==0:
                list_inital[-1]=np.append(list_inital[-1],np.array([action[0]]))

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_obs4[j] = list_obs4
        all_list_act1[j] = list_act1

        #list_inital.append(list_obs1[999])
    # 设置全局字体大小
    plt.rcParams['font.size'] = 18  # 设置默认字体大小为14
    figsize_width = 24
    figsize_height = 12
    # 指定字体文件的完整路径
    font_path = '/usr/share/fonts/truetype/times/times.ttf'
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/times/times.ttf')

    # 创建一个FontProperties对象，指向你的字体文件
    prop = FontProperties(fname=font_path)

    # 更新rcParams以使用指定的字体
    plt.rcParams['font.family'] = prop.get_name()
    print(prop.get_name())
    plt.rcParams['font.style'] = 'normal'

    # for i in range(10):
    #     print(i)
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result1.xlsx',all_list_obs1[i],str(0),i+2) #all_list_obs[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result2.xlsx', all_list_obs2[i], str(0),i+2)  # all_list_obs2[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result3.xlsx', all_list_obs3[i], str(0),i+2)  # all_list_obs3[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result4.xlsx', all_list_obs4[i], str(0),i+2)  # all_list_obs4[i]是
    print(list_inital)
    # 分别为每种观测值绘制10条线
    for k in range(5):  # 对应obs[0], obs[1], obs[2]
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图

        for j in range(10):
            if k == 0:
                data = all_list_obs1[j]
            elif k == 1:
                data = all_list_obs2[j]
            elif k == 2:
                data = all_list_obs3[j]
            elif k == 3:
                data = all_list_obs4[j]
            else:
                data = all_list_act1[j]

            test_data = []
            # if k<=2:
            #     text="Error"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], data,
            #              label=f"Test {j + 1}(inital {text}={round(list_inital[j][k], 1)})")
            # else:
            #     for i in range(len(data)):
            #         test_data.append(data[i]*20)
            #     text="Control Force"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], test_data, label=f"Test {j + 1}(inital {text}={round(list_inital[j][k]*20,1)})")
            if k<=4:
                text="Error"
                num=round(list_inital[k][j],1)
            else:
                num=round(list_inital[k][j],1)*10
                text="Control Force"
            axs.plot([(i / 1000) for i in range(1, len(data) + 1)], data, label=f"Test {j + 1}(inital {text}={num})")
            axs.legend(prop={'size': 11})  # 'upper right'位置参数使图例显示在右上角
        if k==0:
            axs.set_title(f"Error in X")
        elif k==1:
            axs.set_title(f"Error in Y")
        elif k == 2:
            axs.set_title(f"Error in Z")
        elif k==3:
            axs.set_title(f"Error in t")
        elif k==4:
            axs.set_title(f"Control Force of Y")
        axs.set_xlabel("Time(s)")
        if k<=3:
            axs.set_ylabel(f"Error")
        else:
            axs.set_ylabel(f"control term")

        # 显示当前图
        plt.show()
        plt.savefig("training_plot/pt3.png") 
        print("图像已保存为 pt3.png")
        # 关闭图表以释放内存
        plt.close()


def  testyz_revise(model_name0):
    env = gym.make('lorenz_transient-v0')
    print("Environment action space:", env.action_space)
    model = PPO.load(model_name0, env, verbose=1)
    # 创建并保存每种观测值对应的所有线条数据
    # all_list_obs1 = [[] for _ in range(10)]
    # all_list_obs2 = [[] for _ in range(10)]
    # all_list_obs3 = [[] for _ in range(10)]
    # all_list_obs4 = [[] for _ in range(10)]
    # all_list_act1 = [[] for _ in range(10)]

    list_inital = []
    obs = env.reset()
    list_inital=obs[0:4].tolist()
    print(list_inital)

    dx_1_controlled = 30 * (
                2 * list_inital[3] * list_inital[3] * (list_inital[1] - list_inital[0]) + 0.5 *
                list_inital[0])
    dx_2_controlled = 1 * (2 * list_inital[3] * list_inital[3] * (list_inital[0] - list_inital[1]) -
                               list_inital[2])
    dx_3_controlled = 36 * (list_inital[1]- 0.003 * list_inital[2])
    dx_4_controlled = list_inital[1] - list_inital[0] - 0.01 * list_inital[3]

    dx_1_controlled_2 = 30 * (
                    2 * list_inital[3] * list_inital[3] * (list_inital[1] - list_inital[0]) + 0.5 *
                    list_inital[0])
    dx_2_controlled_2 = 1 * (2 * list_inital[3] * list_inital[3] * (list_inital[0] - list_inital[1]) -
                                 list_inital[2])
    dx_3_controlled_2 = 36 * (list_inital[1] - 0.003 * list_inital[2])
    dx_4_controlled_2 = list_inital[1] - list_inital[0] - 0.01 * list_inital[3]

    obs = ([list_inital[0], list_inital[1], list_inital[2], list_inital[3],dx_1_controlled-dx_1_controlled_2, dx_2_controlled-dx_2_controlled_2, dx_3_controlled-dx_3_controlled_2,dx_4_controlled-dx_4_controlled_2])
    list_inital=[]
    list_inital.append(obs[0:4])
    list_obs1, list_obs2, list_obs3,list_obs4 = [], [], [],[]
    list_act1,list_act2,list_act3,list_act4= [],[],[],[]

    for i in range(3000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        list_obs1.append(obs[0])
        list_obs2.append(obs[1])
        list_obs3.append(obs[2])
        list_obs4.append(obs[3])
        list_act1.append(action[0])
        if i==0:
            list_inital[-1]=np.append(list_inital[-1],np.array([action[0]]))

        if i==700:
            print(obs)
        if i==1200:
            print(obs)
        if i==1700:
            print(obs)
    #
    #     # 将每次运行的结果添加到总的列表中
    #     all_list_obs1[j] = list_obs1
    #     all_list_obs2[j] = list_obs2
    #     all_list_obs3[j] = list_obs3
    #     all_list_obs4[j] = list_obs4
    #     all_list_act1[j] = list_act1

    #     #list_inital.append(list_obs1[999])
    # # 设置全局字体大小
    # plt.rcParams['font.size'] = 18  # 设置默认字体大小为14
    # figsize_width = 24
    # figsize_height = 12
    # # 指定字体文件的完整路径
    # font_path = '/usr/share/fonts/truetype/times/times.ttf'
    # font_manager.fontManager.addfont('/usr/share/fonts/truetype/times/times.ttf')
    #
    # # 创建一个FontProperties对象，指向你的字体文件
    # prop = FontProperties(fname=font_path)
    #
    # # 更新rcParams以使用指定的字体
    # plt.rcParams['font.family'] = prop.get_name()
    # print(prop.get_name())
    # plt.rcParams['font.style'] = 'normal'

    # for i in range(10):
    #     print(i)
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result1.xlsx',all_list_obs1[i],str(0),i+2) #all_list_obs[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result2.xlsx', all_list_obs2[i], str(0),i+2)  # all_list_obs2[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result3.xlsx', all_list_obs3[i], str(0),i+2)  # all_list_obs3[i]是
    #     write_to_excel('D:\\考研资料\\江南大学\\组会汇报\\result4.xlsx', all_list_obs4[i], str(0),i+2)  # all_list_obs4[i]是
    # print(list_inital)
    # 分别为每种观测值绘制10条线

def  testyz_pmsm_2(model_name0):
    env = gym.make('lorenz_transient-v0')
    print("Environment action space:", env.action_space)
    model = A2C.load(model_name0, env, verbose=1)
    print("Model action space:", model.action_space)
    print(model.policy)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward/1000:.2f} ± {std_reward/1000:.2f}")

    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(10)]
    all_list_obs2 = [[] for _ in range(10)]
    all_list_obs3 = [[] for _ in range(10)]
    all_list_act1 = [[] for _ in range(10)]


    list_inital=[[4.36934194, -0.4602567,12.2358214,-9.451121,16.03636919,11.92312287,7.67708106,-6.4989595,-2.1164781,-11.94083459],[0.7388987,9.64587271, -0.19743254,10.54901973,0.23867613,-0.40535156,6.51543263,2.84097189,-1.79699836,-7.36938161],[ -3.3901495,-0.11270929,-4.95175868,1.44224794,-6.9494276, -8.77934966,14.72763587,-5.60200807,4.25293769,0.69378025]]

    print(len(list_inital))
    for j in range(10):
        obs=env.reset()

        # dx_1_controlled = -list_inital[0][j] + list_inital[1][j] * list_inital[2][j]
        # dx_2_controlled = -list_inital[1][j] - list_inital[0][j] * list_inital[2][j] + 20 * list_inital[2][j]
        # dx_3_controlled = 5.46 * (list_inital[1][j] - list_inital[2][j])
        #
        # dx_1_controlled_2 = -list_inital[0][j] + list_inital[1][j] * list_inital[2][j]
        # dx_2_controlled_2 = -list_inital[1][j] - list_inital[0][j] * list_inital[2][j] + 20 * list_inital[2][j]
        # dx_3_controlled_2 = 5.46 * (list_inital[1][j] - list_inital[2][j])
        #
        # obs = ([list_inital[0][j], list_inital[1][j], list_inital[2][j],dx_1_controlled-dx_1_controlled_2, dx_2_controlled-dx_2_controlled_2, dx_3_controlled-dx_3_controlled_2])

        list_inital.append(obs[0:3])
        list_obs1, list_obs2, list_obs3 = [], [], []
        list_act1,list_act2,list_act3= [],[],[]

        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_act1.append(action[0])
            if i==0:
                list_inital[-1]=np.append(list_inital[-1],np.array([action[0]]))

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_act1[j] = list_act1

        #list_inital.append(list_obs1[999])
    # 设置全局字体大小
    plt.rcParams['font.size'] = 18  # 设置默认字体大小为14
    figsize_width = 24
    figsize_height = 12
    # 指定字体文件的完整路径
    font_path = '/usr/share/fonts/truetype/times/times.ttf'
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/times/times.ttf')

    # 创建一个FontProperties对象，指向你的字体文件
    prop = FontProperties(fname=font_path)

    # 更新rcParams以使用指定的字体
    plt.rcParams['font.family'] = prop.get_name()
    print(prop.get_name())
    plt.rcParams['font.style'] = 'normal'

    for i in range(10):
        print(i)
        write_to_excel_2('/tmp/pycharm_project_60/code/result1.xlsx',all_list_obs1[i],str(0),i+2) #all_list_obs[i]是
        write_to_excel_2('/tmp/pycharm_project_60/code/result2.xlsx', all_list_obs2[i], str(0),i+2)  # all_list_obs2[i]是
        write_to_excel_2('/tmp/pycharm_project_60/code/result3.xlsx', all_list_obs3[i], str(0),i+2)  # all_list_obs3[i]是
    print(list_inital)
    # 分别为每种观测值绘制10条线
    for k in range(4):  # 对应obs[0], obs[1], obs[2]
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图

        for j in range(10):
            if k == 0:
                data = all_list_obs1[j]
            elif k == 1:
                data = all_list_obs2[j]
            elif k == 2:
                data = all_list_obs3[j]
            else:
                data = all_list_act1[j]

            test_data = []
            # if k<=2:
            #     text="Error"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], data,
            #              label=f"Test {j + 1}(inital {text}={round(list_inital[j][k], 1)})")
            # else:
            #     for i in range(len(data)):
            #         test_data.append(data[i]*20)
            #     text="Control Force"
            #     axs.plot([(i / 100) for i in range(1, len(data) + 1)], test_data, label=f"Test {j + 1}(inital {text}={round(list_inital[j][k]*20,1)})")
            if k<=2:
                text="Error"
                num=round(list_inital[k][j],1)
            # else:
            #     num=round(list_inital[k][j],1)*10
            #     text="Control Force"
            axs.plot([(i / 100) for i in range(1, len(data) + 1)], data, label=f"Test {j + 1}(inital {text}={num})")
            axs.legend(prop={'size': 11})  # 'upper right'位置参数使图例显示在右上角
        if k==0:
            axs.set_title(f"Error in X")
        elif k==1:
            axs.set_title(f"Error in Y")
        elif k == 2:
            axs.set_title(f"Error in Z")
        elif k==3:
            axs.set_title(f"Control Force of Y")
        axs.set_xlabel("Time(s)")
        if k<=2:
            axs.set_ylabel(f"Error")
        else:
            axs.set_ylabel(f"control term")

        # 显示当前图
        plt.show()
        plt.savefig("training_plot/pt4.png") 
        print("图像已保存为 pt4.png")
        # 关闭图表以释放内存
        plt.close()



def testPPO(model_name0,model_name1):
    env = gym.make('lorenz_transient-v0')
    model = SAC.load(model_name0, env, verbose=1)
    model1 = SAC.load(model_name1, env, verbose=1)

    n=10
    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(n)]
    all_list_obs2 = [[] for _ in range(n)]
    all_list_obs3 = [[] for _ in range(n)]
    all_list_act1 = [[] for _ in range(n)]
    all_list_act2 = [[] for _ in range(n)]
    all_list_obs1_2 = [[] for _ in range(n)]
    all_list_obs2_2 = [[] for _ in range(n)]
    all_list_obs3_2 = [[] for _ in range(n)]
    all_list_act1_2 = [[] for _ in range(n)]
    all_list_act2_2 = [[] for _ in range(n)]
    list_inital = []
    list_inital2=[]
    for j in range(n):
        obs = env.reset()
        #list_inital.append(obs[0:3])
        list_obs1, list_obs2, list_obs3 = [], [], []
        list_act1, list_act2 = [], []
        list_obs1_2, list_obs2_2, list_obs3_2 = [], [], []
        list_act1_2, list_act2_2 = [], []

        for i in range(5000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            action_2, _states_2 = model1.predict(obs)
            obs_2, rewards_2, dones_2, info_2 = env.step(action_2)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_act1.append(action[0])
            list_act2.append(action[1])
            list_obs1_2.append(obs_2[0])
            list_obs2_2.append(obs_2[1])
            list_obs3_2.append(obs_2[2])
            list_act1_2.append(action_2[0])
            list_act2_2.append(action_2[1])

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_act1[j] = list_act1
        all_list_act2[j] = list_act2

        all_list_obs1_2[j] = list_obs1_2
        all_list_obs2_2[j] = list_obs2_2
        all_list_obs3_2[j] = list_obs3_2
        all_list_act1_2[j] = list_act1_2
        all_list_act2_2[j] = list_act2_2
        m = 100
        numlist1=[0]*m
        numlist2=[0]*m
        numlist1_y=[0]*m
        numlist2_y = [0] * m
        numlist1_z = [0] * m
        numlist2_z = [0] * m

        for i in range(0,m-1):
            numlist1[i]= numlist1[i]+abs(list_obs1[499-m+i+1])
            numlist2[i] = numlist2[i] + abs(list_obs1_2[499-m + i+1])
            numlist1_y[i] = numlist1_y[i] + abs(list_obs2[499 - m + i + 1])
            numlist2_y[i] = numlist2_y[i] + abs(list_obs2_2[499 - m + i + 1])
            numlist1_z[i] = numlist1_z[i] + abs(list_obs3[499 - m + i + 1])
            numlist2_z[i] = numlist2_z[i] + abs(list_obs3_2[499 - m + i + 1])
        list_inital.append(list_obs1[499])
        list_inital2.append(list_obs1_2[499])
    # 分别为每种观测值绘制10条线
    for k in range(10):  # 对应obs[0], obs[1], obs[2]
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图

        for j in range(n):
            if k == 0:
                data = all_list_obs1[j]
            elif k == 1:
                data = all_list_obs2[j]
            elif k == 2:
                data = all_list_obs3[j]
            elif k == 3:
                data = all_list_act1[j]
            elif k==4:
                data = all_list_act2[j]
            elif k == 5:
                data = all_list_obs1_2[j]
            elif k == 6:
                data = all_list_obs2_2[j]
            elif k == 7:
                data = all_list_obs3_2[j]
            elif k==8:
                data = all_list_act1_2[j]
            else:
                data = all_list_act2_2[j]


            axs.plot(range(1, len(data) + 1), data, label=f"Run {j + 1}")

        axs.set_title(f"Observation {k + 1} Over Steps")
        axs.set_xlabel("Step")
        axs.set_ylabel(f"Observation {k + 1}")
        axs.legend()

        # 显示当前图
        plt.show()
        plt.savefig("training_plot/pt5.png") 
        print("图像已保存为 pt5.png")
        # 关闭图表以释放内存
        plt.close()


    print(list_inital)
    print(list_inital2)
    num1=0
    num2=0
    num3=0
    num4=0
    num5=0
    num6=0
    for i in range(0,m-1):
        num1=num1+numlist1[i]
        num2=num2+numlist2[i]
        num3 = num3 + numlist1_y[i]
        num4 = num4 + numlist2_y[i]
        num5 = num5 + numlist1_z[i]
        num6 = num6 + numlist2_z[i]
    print(num1/m,num2/m,num3/m,num4/m,num5/m,num6/m)

def getResult(model_name0,model_name1):
    env = gym.make('lorenz_transient-v0')
    model = PPO.load(model_name0, env, verbose=1)
    model1 = PPO.load(model_name1, env, verbose=1)

    # list_inital = [[10.91782485, -10.42773029, -28.00696401], [-2.14069824, 17.10024933, 28.56207231]
    #     , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 20.33285682, 6.37122846],
    #                [-3.71330348, -19.18465133, 1.29803615], [19.65005688, -10.10402228, -13.50930468],
    #                [29.22812656, -7.60748863, -27.85328098], [-6.44122029, 19.11934062, 3.28759653],
    #                [-2.05211819, -18.68739753, -19.39912702], [-23.10417921, 7.82553505, -17.04617905],
    #                [14.3, 19.3, -29.9], [-17.7, -0.1, 15.2]
    #     , [29.4, 28.6, 7.1], [19.3, 24.1, -10.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 29.5], [12.6, 27.7, -10.8],
    #                [11.6, -4.8, 18.9], [13.5, 26.9, -5.9]]

    # list_inital = [[10.91782485, -9.42773029, 18.00696401], [-2.14069824, 17.10024933, 18.56207231]
    #     , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 10.33285682, 6.37122846],
    #                [-3.71330348, 19.18465133, 1.29803615], [19.65005688, -9.10402228, 13.50930468],
    #                [12.22812656, -7.60748863, -17.85328098], [-6.44122029, 19.11934062, 3.28759653],
    #                [-2.05211819, -18.68739753, 9.39912702], [13.10417921, 7.82553505, -17.04617905],
    #                [14.3, 19.3, -9.9], [17.7, -0.1, 15.2]
    #     , [12.4, 18.6, 7.1], [19.3, 14.1, -8.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 12.5], [12.6, 17.7, -8.8],
    #                [11.6, -4.8, 18.9], [13.5, 16.9, -5.9]]
    #
    n=20
    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(n)]
    all_list_obs2 = [[] for _ in range(n)]
    all_list_obs3 = [[] for _ in range(n)]
    all_list_obs4 = [[] for _ in range(n)]
    all_list_act1 = [[] for _ in range(n)]
    all_list_act2 = [[] for _ in range(n)]
    all_list_obs1_2 = [[] for _ in range(n)]
    all_list_obs2_2 = [[] for _ in range(n)]
    all_list_obs3_2 = [[] for _ in range(n)]
    all_list_obs4_2 = [[] for _ in range(n)]
    all_list_act1_2 = [[] for _ in range(n)]
    all_list_act2_2 = [[] for _ in range(n)]


    for j in range(n):
        obs = env.reset()
        # dxdt_controlled = -list_inital[j][0] + list_inital[j][1] * list_inital[j][2]
        # dydt_controlled = -list_inital[j][1] - list_inital[j][0] * list_inital[j][2] + 20 * list_inital[j][2]
        # dzdt_controlled = 5.46 * (list_inital[j][1] - list_inital[j][2])
        # obs=([list_inital[j][0],list_inital[j][1],list_inital[j][2],dxdt_controlled,dydt_controlled,dzdt_controlled])
        list_obs1, list_obs2, list_obs3,list_obs4 = [], [], [],[]
        list_act1, list_act2 = [], []
        list_obs1_2, list_obs2_2, list_obs3_2,list_obs4_2 = [], [], [],[]
        list_act1_2, list_act2_2 = [], []


        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            action_2, _states_2 = model1.predict(obs)
            obs_2, rewards_2, dones_2, info_2 = env.step(action_2)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_obs4.append(obs[3])
            list_act1.append(action[0])
            list_act2.append(action[1])
            list_obs1_2.append(obs_2[0])
            list_obs2_2.append(obs_2[1])
            list_obs3_2.append(obs_2[2])
            list_obs4_2.append(obs_2[3])
            list_act1_2.append(action_2[0])
            list_act2_2.append(action_2[1])

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_obs4[j] = list_obs4
        all_list_act1[j] = list_act1
        all_list_act2[j] = list_act2

        all_list_obs1_2[j] = list_obs1_2
        all_list_obs2_2[j] = list_obs2_2
        all_list_obs3_2[j] = list_obs3_2
        all_list_obs4_2[j] = list_obs4_2
        all_list_act1_2[j] = list_act1_2
        all_list_act2_2[j] = list_act2_2

    list_r = []
    list_r1 = []
    list_r2 = []
    list_r3 = []
    list_r4 = []
    list_r5 = []
    list_res=[]

    list_add=[]
    list_add_2=[]


    last_res=0
    last_res1 = 0
    last_res2 = 0
    last_res_add=0
    last_res3 = 0
    last_res4 = 0
    last_res5 = 0
    last_res_add_2 = 0
    res_mae = 0
    res1_mae = 0
    res2_mae = 0
    res_mae_add = 0
    res3_mae = 0
    res4_mae = 0
    res5_mae = 0
    res_mae_add_2 = 0
    for i in range(1000,2000):
        res = 0
        res1 = 0
        res2 = 0

        res_add=0

        res3=0
        res4=0
        res5=0

        res_add_2 = 0
        res_2 = 0
        res1_2 = 0
        res2_2 = 0
        res3_2 = 0
        res4_2 = 0
        res5_2 = 0

        res_add_3 = 0
        res_add_4 = 0

        for j in range(n):
            res+=abs(all_list_obs1[j][i])
            res1 += abs(all_list_obs1_2[j][i])
            res2 += abs(all_list_obs2[j][i])
            res3 += abs(all_list_obs2_2[j][i])
            res4 += abs(all_list_obs3[j][i])
            res5 += abs(all_list_obs3_2[j][i])
            res_add+=abs(all_list_obs4[j][i])
            res_add_2 += abs(all_list_obs4_2[j][i])
            res_2+= (all_list_obs1[j][i])**2
            res1_2+= (all_list_obs1_2[j][i])**2
            res2_2 += (all_list_obs2[j][i])**2
            res3_2 += (all_list_obs2_2[j][i])**2
            res4_2 += (all_list_obs3[j][i])**2
            res5_2 += (all_list_obs3_2[j][i])**2
            res_add_3 +=(all_list_obs4[j][i])**2
            res_add_4 +=(all_list_obs4_2[j][i])**2
        list_r.append(res/n)
        list_r1.append(res1 / n)
        list_r2.append(res2 / n)
        list_r3.append(res3 / n)
        list_r4.append(res4 / n)
        list_r5.append(res5 / n)

        list_add.append(res_add)
        list_add_2.append(res_add_2)

        res_mae+= res/n
        res1_mae += res2/n
        res2_mae += res4/n
        res3_mae += res1/n
        res4_mae += res3/n
        res5_mae += res5/n
        res_mae_add+=res_add
        res_mae_add_2+=res_add_2


        last_res+=res_2/n
        last_res1+=res2_2/n
        last_res2+=res4_2/n
        last_res3 += res1_2 / n
        last_res4 += res3_2 / n
        last_res5 += res5_2 / n

        last_res_add+=res_add/n
        last_res_add_2+=res_add_2/n


    list_res.append(list_r)
    list_res.append(list_r1)
    list_res.append(list_r2)
    list_res.append(list_r3)
    list_res.append(list_r4)
    list_res.append(list_r5)
    list_res.append(list_add)
    list_res.append(list_add_2)

    last_res = math.sqrt(last_res/1000)
    last_res1 = math.sqrt(last_res1/1000)
    last_res2 = math.sqrt(last_res2/1000)
    last_res3 = math.sqrt(last_res3/1000)
    last_res4 = math.sqrt(last_res4/1000)
    last_res5 = math.sqrt(last_res5/1000)
    last_res_add = math.sqrt(last_res_add/1000)
    last_res_add_2 = math.sqrt(last_res_add_2/1000)

    print(res_mae/1000, res1_mae/1000, res2_mae/1000, res_mae_add/1000, (res_mae/1000 + res1_mae/1000 + res2_mae/1000+res_mae_add/1000) / 4)
    print(res3_mae/1000, res4_mae/1000, res5_mae/1000, res_mae_add_2/1000,
          (res3_mae/1000 + res4_mae/1000 + res5_mae/1000+res_mae_add_2/1000) / 4)

    print(last_res,last_res1,last_res2,last_res_add,(last_res+last_res1+last_res2+last_res_add)/4)
    print(last_res3 , last_res4 , last_res5 , last_res_add_2,
          (last_res3  + last_res4  + last_res5 +last_res_add) / 4)

    # for k in range(3):
    #     fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图
    #     for i in range(2):
    #         data = list_res[(i+1)*(k+1)-1]
    #         axs.plot(range(1, len(data) + 1), data, label=f"Run {j + 1}")
    #
    #     axs.set_title(f"Observation {k + 1} Over Steps")
    #     axs.set_xlabel("Step")
    #     axs.set_ylabel(f"Observation {k + 1}")
    #     axs.legend()
    #     # 显示当前图
    #     plt.show()


def getResult_pmsm(model_name0,model_name1):
    env = gym.make('lorenz_transient-v0')
    model = A2C.load(model_name0, env, verbose=1)
    model1 = A2C.load(model_name1, env, verbose=1)

    # list_inital = [[10.91782485, -10.42773029, -28.00696401], [-2.14069824, 17.10024933, 28.56207231]
    #     , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 20.33285682, 6.37122846],
    #                [-3.71330348, -19.18465133, 1.29803615], [19.65005688, -10.10402228, -13.50930468],
    #                [29.22812656, -7.60748863, -27.85328098], [-6.44122029, 19.11934062, 3.28759653],
    #                [-2.05211819, -18.68739753, -19.39912702], [-23.10417921, 7.82553505, -17.04617905],
    #                [14.3, 19.3, -29.9], [-17.7, -0.1, 15.2]
    #     , [29.4, 28.6, 7.1], [19.3, 24.1, -10.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 29.5], [12.6, 27.7, -10.8],
    #                [11.6, -4.8, 18.9], [13.5, 26.9, -5.9]]

    # list_inital = [[10.91782485, -9.42773029, 18.00696401], [-2.14069824, 17.10024933, 18.56207231]
    #     , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 10.33285682, 6.37122846],
    #                [-3.71330348, 19.18465133, 1.29803615], [19.65005688, -9.10402228, 13.50930468],
    #                [12.22812656, -7.60748863, -17.85328098], [-6.44122029, 19.11934062, 3.28759653],
    #                [-2.05211819, -18.68739753, 9.39912702], [13.10417921, 7.82553505, -17.04617905],
    #                [14.3, 19.3, -9.9], [17.7, -0.1, 15.2]
    #     , [12.4, 18.6, 7.1], [19.3, 14.1, -8.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 12.5], [12.6, 17.7, -8.8],
    #                [11.6, -4.8, 18.9], [13.5, 16.9, -5.9]]
    #
    n=20
    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(n)]
    all_list_obs2 = [[] for _ in range(n)]
    all_list_obs3 = [[] for _ in range(n)]
    all_list_act1 = [[] for _ in range(n)]
    all_list_act2 = [[] for _ in range(n)]
    all_list_obs1_2 = [[] for _ in range(n)]
    all_list_obs2_2 = [[] for _ in range(n)]
    all_list_obs3_2 = [[] for _ in range(n)]
    all_list_act1_2 = [[] for _ in range(n)]
    all_list_act2_2 = [[] for _ in range(n)]


    for j in range(n):
        obs = env.reset()
        # dxdt_controlled = -list_inital[j][0] + list_inital[j][1] * list_inital[j][2]
        # dydt_controlled = -list_inital[j][1] - list_inital[j][0] * list_inital[j][2] + 20 * list_inital[j][2]
        # dzdt_controlled = 5.46 * (list_inital[j][1] - list_inital[j][2])
        # obs=([list_inital[j][0],list_inital[j][1],list_inital[j][2],dxdt_controlled,dydt_controlled,dzdt_controlled])
        list_obs1, list_obs2, list_obs3 = [], [], []
        list_act1, list_act2 = [], []
        list_obs1_2, list_obs2_2, list_obs3_2= [], [], []
        list_act1_2, list_act2_2 = [], []


        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            action_2, _states_2 = model1.predict(obs)
            obs_2, rewards_2, dones_2, info_2 = env.step(action_2)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_act1.append(action[0])
            list_act2.append(action[1])
            list_obs1_2.append(obs_2[0])
            list_obs2_2.append(obs_2[1])
            list_obs3_2.append(obs_2[2])
            list_act1_2.append(action_2[0])
            list_act2_2.append(action_2[1])

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_act1[j] = list_act1
        all_list_act2[j] = list_act2

        all_list_obs1_2[j] = list_obs1_2
        all_list_obs2_2[j] = list_obs2_2
        all_list_obs3_2[j] = list_obs3_2
        all_list_act1_2[j] = list_act1_2
        all_list_act2_2[j] = list_act2_2

    list_r = []
    list_r1 = []
    list_r2 = []
    list_r3 = []
    list_r4 = []
    list_r5 = []
    list_res=[]

    list_add=[]
    list_add_2=[]


    last_res=0
    last_res1 = 0
    last_res2 = 0

    last_res3 = 0
    last_res4 = 0
    last_res5 = 0

    res_mae = 0
    res1_mae = 0
    res2_mae = 0

    res3_mae = 0
    res4_mae = 0
    res5_mae = 0

    for i in range(1000,2000):
        res = 0
        res1 = 0
        res2 = 0

        res_add=0

        res3=0
        res4=0
        res5=0

        res_add_2 = 0
        res_2 = 0
        res1_2 = 0
        res2_2 = 0
        res3_2 = 0
        res4_2 = 0
        res5_2 = 0

        res_add_3 = 0
        res_add_4 = 0

        for j in range(n):
            res+=abs(all_list_obs1[j][i])
            res1 += abs(all_list_obs1_2[j][i])
            res2 += abs(all_list_obs2[j][i])
            res3 += abs(all_list_obs2_2[j][i])
            res4 += abs(all_list_obs3[j][i])
            res5 += abs(all_list_obs3_2[j][i])
            res_2+= (all_list_obs1[j][i])**2
            res1_2+= (all_list_obs1_2[j][i])**2
            res2_2 += (all_list_obs2[j][i])**2
            res3_2 += (all_list_obs2_2[j][i])**2
            res4_2 += (all_list_obs3[j][i])**2
            res5_2 += (all_list_obs3_2[j][i])**2
        list_r.append(res/n)
        list_r1.append(res1 / n)
        list_r2.append(res2 / n)
        list_r3.append(res3 / n)
        list_r4.append(res4 / n)
        list_r5.append(res5 / n)


        res_mae+= res/n
        res1_mae += res2/n
        res2_mae += res4/n
        res3_mae += res1/n
        res4_mae += res3/n
        res5_mae += res5/n



        last_res+=res_2/n
        last_res1+=res2_2/n
        last_res2+=res4_2/n
        last_res3 += res1_2 / n
        last_res4 += res3_2 / n
        last_res5 += res5_2 / n

    list_res.append(list_r)
    list_res.append(list_r1)
    list_res.append(list_r2)
    list_res.append(list_r3)
    list_res.append(list_r4)
    list_res.append(list_r5)
    list_res.append(list_add)
    list_res.append(list_add_2)

    last_res = math.sqrt(last_res/1000)
    last_res1 = math.sqrt(last_res1/1000)
    last_res2 = math.sqrt(last_res2/1000)
    last_res3 = math.sqrt(last_res3/1000)
    last_res4 = math.sqrt(last_res4/1000)
    last_res5 = math.sqrt(last_res5/1000)

    print(res_mae/1000, res1_mae/1000, res2_mae/1000, (res_mae/1000 + res1_mae/1000 + res2_mae/1000) / 3)
    print(res3_mae/1000, res4_mae/1000, res5_mae/1000,
          (res3_mae/1000 + res4_mae/1000 + res5_mae/1000) / 3)

    print(last_res,last_res1,last_res2,(last_res+last_res1+last_res2)/3)
    print(last_res3 , last_res4 , last_res5,
          (last_res3  + last_res4  + last_res5) / 3)

    # for k in range(3):
    #     fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图
    #     for i in range(2):
    #         data = list_res[(i+1)*(k+1)-1]
    #         axs.plot(range(1, len(data) + 1), data, label=f"Run {j + 1}")
    #
    #     axs.set_title(f"Observation {k + 1} Over Steps")
    #     axs.set_xlabel("Step")
    #     axs.set_ylabel(f"Observation {k + 1}")
    #     axs.legend()
    #     # 显示当前图
    #     plt.show()



def getResult2(model_name0,model_name1):
    env = gym.make('lorenz_transient-v0')
    model = PPO.load(model_name0, env, verbose=1)
    model1 = PPO.load(model_name1, env, verbose=1)

    # list_inital = [[10.91782485, -10.42773029, -28.00696401], [-2.14069824, 17.10024933, 28.56207231]
    #     , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 20.33285682, 6.37122846],
    #                [-3.71330348, -19.18465133, 1.29803615], [19.65005688, -10.10402228, -13.50930468],
    #                [29.22812656, -7.60748863, -27.85328098], [-6.44122029, 19.11934062, 3.28759653],
    #                [-2.05211819, -18.68739753, -19.39912702], [-23.10417921, 7.82553505, -17.04617905],
    #                [14.3, 19.3, -29.9], [-17.7, -0.1, 15.2]
    #     , [29.4, 28.6, 7.1], [19.3, 24.1, -10.3],
    #                [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
    #                [4.4, 0.2, 29.5], [12.6, 27.7, -10.8],
    #                [11.6, -4.8, 18.9], [13.5, 26.9, -5.9]]

    list_inital = [[10.91782485, -9.42773029, 18.00696401], [-2.14069824, 17.10024933, 18.56207231]
        , [-2.8781161, -7.7740351, -9.83865214], [-9.00023262, 10.33285682, 6.37122846],
                   [-3.71330348, 19.18465133, 1.29803615], [19.65005688, -9.10402228, 13.50930468],
                   [12.22812656, -7.60748863, -17.85328098], [-6.44122029, 19.11934062, 3.28759653],
                   [-2.05211819, -18.68739753, 9.39912702], [13.10417921, 7.82553505, -17.04617905],
                   [14.3, 19.3, -9.9], [17.7, -0.1, 15.2]
        , [12.4, 18.6, 7.1], [19.3, 14.1, -8.3],
                   [-4.8, 15.5, -2.0], [14.1, 2.4, -16.5],
                   [4.4, 0.2, 12.5], [12.6, 17.7, -8.8],
                   [11.6, -4.8, 18.9], [13.5, 16.9, -5.9]]

    n=20
    # 创建并保存每种观测值对应的所有线条数据
    all_list_obs1 = [[] for _ in range(n)]
    all_list_obs2 = [[] for _ in range(n)]
    all_list_obs3 = [[] for _ in range(n)]
    all_list_act1 = [[] for _ in range(n)]
    all_list_act2 = [[] for _ in range(n)]
    all_list_obs1_2 = [[] for _ in range(n)]
    all_list_obs2_2 = [[] for _ in range(n)]
    all_list_obs3_2 = [[] for _ in range(n)]
    all_list_act1_2 = [[] for _ in range(n)]
    all_list_act2_2 = [[] for _ in range(n)]


    for j in range(n):
        obs = env.reset()
        dxdt_controlled = -list_inital[j][0] + list_inital[j][1] * list_inital[j][2]
        dydt_controlled = -list_inital[j][1] - list_inital[j][0] * list_inital[j][2] + 20 * list_inital[j][2]
        dzdt_controlled = 5.46 * (list_inital[j][1] - list_inital[j][2])
        obs=([list_inital[j][0],list_inital[j][1],list_inital[j][2]])
        list_obs1, list_obs2, list_obs3 = [], [], []
        list_act1, list_act2 = [], []
        list_obs1_2, list_obs2_2, list_obs3_2 = [], [], []
        list_act1_2, list_act2_2 = [], []


        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            action_2, _states_2 = model1.predict(obs)
            obs_2, rewards_2, dones_2, info_2 = env.step(action_2)
            list_obs1.append(obs[0])
            list_obs2.append(obs[1])
            list_obs3.append(obs[2])
            list_act1.append(action[0])
            list_act2.append(action[1])
            list_obs1_2.append(obs_2[0])
            list_obs2_2.append(obs_2[1])
            list_obs3_2.append(obs_2[2])
            list_act1_2.append(action_2[0])
            list_act2_2.append(action_2[1])

        # 将每次运行的结果添加到总的列表中
        all_list_obs1[j] = list_obs1
        all_list_obs2[j] = list_obs2
        all_list_obs3[j] = list_obs3
        all_list_act1[j] = list_act1
        all_list_act2[j] = list_act2

        all_list_obs1_2[j] = list_obs1_2
        all_list_obs2_2[j] = list_obs2_2
        all_list_obs3_2[j] = list_obs3_2
        all_list_act1_2[j] = list_act1_2
        all_list_act2_2[j] = list_act2_2

    list_r = []
    list_r1 = []
    list_r2 = []
    list_r3 = []
    list_r4 = []
    list_r5 = []
    list_res=[]


    last_res=0
    last_res1 = 0
    last_res2 = 0
    last_res3 = 0
    last_res4 = 0
    last_res5 = 0
    res_mae = 0
    res1_mae = 0
    res2_mae = 0
    res3_mae = 0
    res4_mae = 0
    res5_mae = 0
    for i in range(1000,2000):
        res = 0
        res1 = 0
        res2 = 0
        res3=0
        res4=0
        res5=0
        res_2 = 0
        res1_2 = 0
        res2_2 = 0
        res3_2 = 0
        res4_2 = 0
        res5_2 = 0
        for j in range(n):
            res+=abs(all_list_obs1[j][i])
            res1 += abs(all_list_obs1_2[j][i])
            res2 += abs(all_list_obs2[j][i])
            res3 += abs(all_list_obs2_2[j][i])
            res4 += abs(all_list_obs3[j][i])
            res5 += abs(all_list_obs3_2[j][i])
            res_2+= (all_list_obs1[j][i])**2
            res1_2+= (all_list_obs1_2[j][i])**2
            res2_2 += (all_list_obs2[j][i])**2
            res3_2 += (all_list_obs2_2[j][i])**2
            res4_2 += (all_list_obs3[j][i])**2
            res5_2 += (all_list_obs3_2[j][i])**2
        list_r.append(res/n)
        list_r1.append(res1 / n)
        list_r2.append(res2 / n)
        list_r3.append(res3 / n)
        list_r4.append(res4 / n)
        list_r5.append(res5 / n)


        res_mae+= res/n
        res1_mae += res2/n
        res2_mae += res4/n
        res3_mae += res1/n
        res4_mae += res3/n
        res5_mae += res5/n

        last_res+=res_2/n
        last_res1+=res2_2/n
        last_res2+=res4_2/n
        last_res3 += res1_2 / n
        last_res4 += res3_2 / n
        last_res5 += res5_2 / n



    list_res.append(list_r)
    list_res.append(list_r1)
    list_res.append(list_r2)
    list_res.append(list_r3)
    list_res.append(list_r4)
    list_res.append(list_r5)

    last_res = math.sqrt(last_res/1000)
    last_res1 = math.sqrt(last_res1/1000)
    last_res2 = math.sqrt(last_res2/1000)
    last_res3 = math.sqrt(last_res3/1000)
    last_res4 = math.sqrt(last_res4/1000)
    last_res5 = math.sqrt(last_res5/1000)

    print(res_mae/1000, res1_mae/1000, res2_mae/1000, (res_mae/1000 + res1_mae/1000 + res2_mae/1000) / 3)
    print(res3_mae/1000, res4_mae/1000, res5_mae/1000,
          (res3_mae/1000 + res4_mae/1000 + res5_mae/1000) / 3)

    print(last_res,last_res1,last_res2,(last_res+last_res1+last_res2)/3)
    print(last_res3 , last_res4 , last_res5 ,
          (last_res3  + last_res4  + last_res5 ) / 3)

    for k in range(3):
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))  # 每次只创建一张图
        for i in range(2):
            data = list_res[(i+1)*(k+1)-1]
            axs.plot(range(1, len(data) + 1), data, label=f"Run {j + 1}")

        axs.set_title(f"Observation {k + 1} Over Steps")
        axs.set_xlabel("Step")
        axs.set_ylabel(f"Observation {k + 1}")
        axs.legend()
        # 显示当前图
        plt.show()
        plt.savefig("training_plot/pt6.png") 
        print("图像已保存为 pt6.png")
        # 关闭图表以释放内存
        plt.close()

def write_data_to_file(all_list_act1, all_list_act2, filename):
    with open(filename, 'w') as file:
        # 写入act1的数据
        file.write("# act1\n")
        for i, sublist in enumerate(all_list_act1):
            if i > 0:  # 在非第一个子列表前加空行
                file.write("\n")
            line = ",".join(map(str, sublist))  # 将子列表转换为字符串，并用逗号分隔
            file.write(line + "\n")

        # 写入act2的数据
        file.write("\n# act2\n")
        for i, sublist in enumerate(all_list_act2):
            if i > 0:  # 在非第一个子列表前加空行
                file.write("\n")
            line = ",".join(map(str, sublist))  # 将子列表转换为字符串，并用逗号分隔
            file.write(line + "\n")


def calculate_root_sum_of_squares(file_path, sheet_index=0, column_index=0):
    """
    读取指定 Excel 工作表中的某一列（从第 1000 行到第 2999 行），
    计算这些数值的平方和，然后对结果开平方根。

    参数:
    - file_path: Excel 文件的路径。
    - sheet_index: 要读取的工作表索引，默认为 0（第一个工作表）。
    - column_index: 要读取的列索引，默认为 0（第一列）。

    返回:
    - 平方和的平方根。
    """
    # 读取 Excel 文件中的指定工作表和列
    df = pd.read_excel(file_path, sheet_name=sheet_index, usecols=[column_index], skiprows=999, nrows=2000, header=None)

    # 提取数据并转换为 NumPy 数组
    data = df.iloc[:, 0].values  # 使用 iloc 确保我们总是获取正确的列

    # 计算每一行数字的平方
    squared_data = np.square(data)

    # 计算平方和
    sum_of_squares = np.sum(squared_data)

    # 对平方和开根号
    result = np.sqrt(sum_of_squares / 2000)  # 注意：这里除以2000是因为你之前的代码中有这个操作

    return result


def calculate_root_sum_of_squares2(file_path, sheet_index=0, column_index=0):
    """
    读取指定 Excel 工作表中的某一列（从第 1000 行到第 2999 行），
    计算这些数值的平方和，然后对结果开平方根。

    参数:
    - file_path: Excel 文件的路径。
    - sheet_index: 要读取的工作表索引，默认为 0（第一个工作表）。
    - column_index: 要读取的列索引，默认为 0（第一列）。

    返回:
    - 平方和的平方根。
    """
    # 读取 Excel 文件中的指定工作表和列
    df = pd.read_excel(file_path, sheet_name=sheet_index, usecols=[column_index], skiprows=999, nrows=1000, header=None)

    # 提取数据并转换为 NumPy 数组
    data = df.iloc[:, 0].values  # 使用 iloc 确保我们总是获取正确的列


    abs_data = np.abs(data)


    sum_of_squares = np.sum(abs_data)

    # 对平方和开根号
    result = np.sqrt(sum_of_squares / 1000)  # 注意：这里除以2000是因为你之前的代码中有这个操作

    return result


if __name__=='__main__':

    save_model_name = 'lorenz_f2_lr5en5_s1m'
    # file_path = '/tmp/pycharm_project_60/code/chaos_apl/mae,mse.xlsx'
    # result=0
    # results=[]
    # for j in range(8):
    #     for i in range(10):
    #         print(i)
    #         result = result+calculate_root_sum_of_squares2(file_path, sheet_index=j, column_index=i)
    #
    #     print(f"The root of the sum of squares is: {result/10}")
    #     results.append(result/10)
    # callback = function(save_model_name)
    #testyz('lorenz_targeting_Lstm_continous_0')

    # 开始计时
    # start_time = time.time()
    #
    # # 调用函数
    # testyz_revise('lorenz_targeting_Lstm_continous_0')
    #
    # # 结束计时
    # end_time = time.time()
    #
    # # 输出耗时
    # print(f"testyz() 函数执行耗时: {end_time - start_time:.4f} 秒")

    #testyz_pmsm_2('lorenz_targeting_Lstm_continous_0')

    testPPO1('lorenz_f2_lr5en5_s1m')



    #testPPO3('lorenz_targeting_Lstm_continous_0')
    #testPPO('lorenz_targeting_Lstm_continous_0','lorenz_targeting_Lstm_continous_1')
    #getResult_pmsm('lorenz_targeting_Lstm_continous_0', 'lorenz_targeting_Lstm_continous_0')
    #getResult('lorenz_targeting_Lstm_continous_0','lorenz_targeting_Lstm_continous_0')
    #getResult2('lorenz_targeting_Lstm_continous_0', 'lorenz_targeting_Lstm_continous_0')
