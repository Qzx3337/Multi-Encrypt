import os

import matplotlib.pyplot as plt
#绘制三维图像
import mpl_toolkits.mplot3d as p3d
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
# from lorenz_singlecontrol import lorenzEnv_transient, lorenzEnv_transient_hr, lorenzEnv_transient_pmsm
from lorenz_singlecontrol import lorenzEnv_transient


import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FuncFormatter

def thousandth_formatter(x, pos):
    return '%d' % int(x / 1000)

# sys.path.append('/tmp/pycharm_project_60/code/gym-lorenz/gym_lorenz/envs/')  # 或者确切的安装路径
#
#
# font_path = '/usr/share/fonts/truetype/times/times.ttf'
# font_manager.fontManager.addfont('/usr/share/fonts/truetype/times/times.ttf')
# # 创建一个FontProperties对象，指向你的字体文件
# prop = FontProperties(fname=font_path)
# # 更新rcParams以使用指定的字体
# plt.rcParams['font.family'] = prop.get_name()
# plt.rcParams['font.style'] = 'normal'

def test():
  env=lorenzEnv_transient()
  x=[]
  y=[]
  z=[]
  env.reset()
  for i in range(10000):
      obs, rewards, dones, info = env.step()
      x.append(obs[0])
      y.append(obs[1])
      z.append(obs[2])
  fig = plt.figure(figsize=(7, 6))
  ax = fig.add_subplot(111, projection='3d')
  fig.subplots_adjust(left=0, right=0.9, bottom=0.1, top=1)

  for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
      axis.label.set_fontsize(20)  # 设置坐标轴标签字体大小
      axis.labelpad = 20 # 增加坐标轴标签与轴的距离，可选
      axis.set_tick_params(labelsize=20)  # 设置刻度标签字体大小



  # 绘制轨迹
  ax.plot(x, y, z, lw=0.5, color='blue')

  # 美化图表
  ax.set_xlabel('x Axis', fontsize=25)
  ax.set_ylabel('y Axis', fontsize=25)
  ax.set_zlabel('z Axis', fontsize=25)
  # ax.view_init(elev=30, azim=45)  # 设置视角
  ax.grid(True)

  # 显示图形
  plt.show()

def test_pmsm():
  env=lorenzEnv_transient_pmsm()
  x=[]
  y=[]
  z=[]
  env.reset()
  for i in range(20000):
      obs, rewards, dones, info = env.step()
      x.append(obs[0])
      y.append(obs[1])
      z.append(obs[2])
  fig = plt.figure(figsize=(7, 6))
  ax = fig.add_subplot(111, projection='3d')
  fig.subplots_adjust(left=0, right=0.9, bottom=0.1, top=1)

  for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
      axis.label.set_fontsize(20)  # 设置坐标轴标签字体大小
      axis.labelpad = 20 # 增加坐标轴标签与轴的距离，可选
      axis.set_tick_params(labelsize=20)  # 设置刻度标签字体大小



  # 绘制轨迹
  ax.plot(x, y, z, lw=0.5, color='blue')

  # 美化图表
  ax.set_xlabel(r'$i_d$', fontsize=25)
  ax.set_ylabel(r'$i_q$', fontsize=25)
  ax.set_zlabel(r'$\omega$', fontsize=25)
  # ax.view_init(elev=30, azim=45)  # 设置视角
  ax.grid(True)

  # 显示图形
  plt.show()

import time

def testyz():
    env = lorenzEnv_transient()  # 假设这是你的环境设置函数
    x = []
    y = []
    z = []
    t = []

    env.reset()
    for i in range(5000):
        obs, rewards, dones, info = env.step()
        x.append(obs[0])
        y.append(obs[1])
        z.append(obs[2])
        t.append(obs[3])


    # # 绘制四条曲线
    # figs = [plt.figure(figsize=(15, 4)) for _ in range(4)]
    # axes = [fig.add_subplot(111) for fig in figs]
    # data_sets = [(x, 'x'), (y, 'y'), (z, 'z'), (t, 't')]
    #
    # for ax, data, label in zip(axes, data_sets, ['x', 'y', 'z', 't']):
    #     ax.plot(data[0])  # 数据绘图
    #     ax.set_ylabel(label, fontsize=15)
    #     ax.xaxis.set_major_formatter(FuncFormatter(thousandth_formatter))
    #     ax.set_xlabel('Time(s)', fontsize=15)
    #
    #     # 设置x轴的起始值为数据中的最小值或0，使得零刻度线靠近左边
    #     if min(data[0]) < 0:
    #         start = min(data[0])
    #     else:
    #         start = 0
    #     ax.set_xlim(left=start)
    # write_to_excel('/tmp/pycharm_project_60/code/ori_mcs.xlsx', x, y, z,
    #                t, "1")
    # plt.show()


def testyz4():
    env = lorenzEnv_transient()  # 假设这是你的环境设置函数
    x = []
    y = []
    z = []


    env.reset()
    for i in range(100000):
        obs, rewards, dones, info = env.step()
        x.append(obs[0])
        y.append(obs[1])
        z.append(obs[2])

    # 绘制四条曲线
    figs = [plt.figure(figsize=(15, 4)) for _ in range(3)]
    axes = [fig.add_subplot(111) for fig in figs]
    data_sets = [(x, 'x'), (y, 'y'), (z, 'z')]

    for ax, data, label in zip(axes, data_sets, ['x', 'y', 'z']):
        ax.plot(data[0])  # 数据绘图
        ax.set_ylabel(label, fontsize=15)
        ax.xaxis.set_major_formatter(FuncFormatter(thousandth_formatter))
        ax.set_xlabel('Time(s)', fontsize=15)

        # 设置x轴的起始值为数据中的最小值或0，使得零刻度线靠近左边
        if min(data[0]) < 0:
            start = min(data[0])
        else:
            start = 0
        ax.set_xlim(left=start)
    write_to_excel('/tmp/pycharm_project_60/code/result_lorenz.xlsx', x, y, z,
                    "1")
    plt.show()


def write_to_excel(filepath, x, y, z, sheetName):
    # 确保所有列表的长度相同
    if not (len(x) == len(y) == len(z)):
        raise ValueError("All input lists must have the same length.")

    # 生成 time 列，从 0 开始，每行增加 0.001
    time = [i * 0.01 for i in range(len(x))]

    # 创建DataFrame，列名分别为1, 2, 3, 4, 5（time）
    df = pd.DataFrame({
        1: x,
        2: y,
        3: z,
        4: time
    })

    if os.path.exists(filepath):
        try:
            # 尝试以追加模式打开现有Excel文件
            with pd.ExcelWriter(filepath, mode='a', engine='openpyxl') as writer:
                book = writer.book
                if sheetName in book.sheetnames:
                    # 如果sheet已存在，删除旧的sheet以避免错误
                    sheet = book[sheetName]
                    book.remove(sheet)
                df.to_excel(writer, sheet_name=sheetName, index=False)
        except Exception as e:
            print(f"Error opening existing Excel file: {e}. Creating new file.")
            # 如果文件不是一个有效的Excel文件，重新创建
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheetName, index=False)
    else:
        # 文件不存在，直接写入并创建新的sheet
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheetName, index=False)


# def write_to_excel(filepath, x, y, z, t, sheetName):
#     # 确保所有列表的长度相同
#     if not (len(x) == len(y) == len(z) == len(t)):
#         raise ValueError("All input lists must have the same length.")
#
#     # 生成 time 列，从 0 开始，每行增加 0.001
#     time = [i * 0.001 for i in range(len(x))]
#
#     # 创建DataFrame，列名分别为1, 2, 3, 4, 5（time）
#     df = pd.DataFrame({
#         1: x,
#         2: y,
#         3: z,
#         4: t,
#         5: time
#     })
#
#     if os.path.exists(filepath):
#         try:
#             # 尝试以追加模式打开现有Excel文件
#             with pd.ExcelWriter(filepath, mode='a', engine='openpyxl') as writer:
#                 book = writer.book
#                 if sheetName in book.sheetnames:
#                     # 如果sheet已存在，删除旧的sheet以避免错误
#                     sheet = book[sheetName]
#                     book.remove(sheet)
#                 df.to_excel(writer, sheet_name=sheetName, index=False)
#         except Exception as e:
#             print(f"Error opening existing Excel file: {e}. Creating new file.")
#             # 如果文件不是一个有效的Excel文件，重新创建
#             with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
#                 df.to_excel(writer, sheet_name=sheetName, index=False)
#     else:
#         # 文件不存在，直接写入并创建新的sheet
#         with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name=sheetName, index=False)

# def gettest(filepath):
#     # CSV文件路径
#     csv_file_path = filepath
#
#     # 使用pandas读取CSV文件
#     data = pd.read_csv(csv_file_path)
#
#     # 假设CSV文件中的第一列是x轴，第二列是y轴
#     x = data.iloc[:, 1]
#     y = data.iloc[:, 2]
#
#     # 使用matplotlib生成图片
#     plt.figure(figsize=(10, 6))  # 设置图片大小
#
#     x_np = x.to_numpy()
#     y_np = y.to_numpy()
#
#     plt.plot(x_np, y_np, color='blue')
#
#     plt.xlabel('step')
#     plt.ylabel('value')
#
#     # 添加图例
#     plt.legend()
#
#     # 显示图形
#     plt.show()


def testyz2():


    env = lorenzEnv_transient()  # 假设这是你的环境设置函数
    x = []
    z = []

    env.reset()
    for i in range(1000000):
        obs, rewards, dones, info = env.step()
        x.append(obs[0])
        z.append(obs[2])

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 x-z 散点图
    ax.scatter(x, z)

    # 设置标题和轴标签
    ax.set_title('hr Attractor x-z Scatter Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    # 显示图形
    plt.show()


def simulate_system(n_steps=500000):
    # 假设 lorenzEnv_transient 是一个可以模拟 Lorenz 系统的环境
    env = lorenzEnv_transient()
    x = []
    y = []
    z = []
    t = []
    env.reset()
    for i in range(n_steps):
        obs, rewards, dones, info = env.step()
        x.append(obs[0])
        y.append(obs[1])
        z.append(obs[2])
        t.append(obs[3])

    return np.array(x), np.array(y), np.array(z), np.array(t)


def plot_single_phase(var_x, var_y, var_z, labels, title, fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制数据
    ax.plot(var_x, var_y, var_z, lw=0.5)

    # 设置标签和标题
    # ax.set_xlabel(labels[0], fontsize=15)
    # ax.set_ylabel(labels[1], fontsize=15)
    # ax.set_zlabel(labels[2], fontsize=15)
    ax.set_title(title, fontsize=15)

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    # 调整子图位置，使其占满整个图片
    plt.tight_layout()

    # 设置坐标轴范围
    ax.set_xlim([min(var_x), max(var_x)])
    ax.set_ylim([min(var_y), max(var_y)])
    ax.set_zlim([min(var_z), max(var_z)])

    # 设置x、y和z轴刻度的字体大小
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontsize(20)  # 设置你想要的字体大小

    # ax.tick_params(axis='x', which='major', pad=7)  # 增加x轴刻度标签与刻度线之间的距离
    # ax.tick_params(axis='y', which='major', pad=7)  # 增加y轴刻度标签与刻度线之间的距离
    ax.tick_params(axis='z', which='major', pad=12)  # 增加z轴刻度标签与刻度线之间的距离

    # 调整相机距离
    ax.dist = 10

    # 调整坐标轴标签的位置
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    plt.show()
if __name__=='__main__':
    #test_pmsm()
    #testyz()
    # 开始计时
    start_time = time.time()

    # 调用函数
    testyz()

    # 结束计时
    end_time = time.time()

    # 输出耗时
    print(f"testyz() 函数执行耗时: {end_time - start_time:.4f} 秒")
    # # 模拟系统生成数据
    # x, y, z, t = simulate_system()
    #
    # # 定义每张图的标签和标题以及视角参数
    # plots_info = [
    #     ((r'$x_1(t)$', r'$x_2(t)$', r'$x_3(t)$'), r'$x_3(t)$ vs $x_1(t)$ vs $x_2(t)$', 1, 30, -60),
    #     ((r'$x_1(t)$', r'$x_2(t)$', r'$x_4(t)$'), r'$x_4(t)$ vs $x_1(t)$ vs $x_2(t)$', 2, 30, -60),
    #     ((r'$x_1(t)$', r'$x_3(t)$', r'$x_4(t)$'), r'$x_4(t)$ vs $x_1(t)$ vs $x_3(t)$', 3, 30, -60),
    #     ((r'$x_2(t)$', r'$x_3(t)$', r'$x_4(t)$'), r'$x_4(t)$ vs $x_2(t)$ vs $x_3(t)$', 4, 30, -60),
    # ]
    #
    # for idx, (labels, title, fig_num, elev, azim) in enumerate(plots_info):
    #     vars_dict = {r'$x_1(t)$': x, r'$x_2(t)$': y, r'$x_3(t)$': z, r'$x_4(t)$': t}
    #     var_x = vars_dict[labels[0]]
    #     var_y = vars_dict[labels[1]]
    #     var_z = vars_dict[labels[2]]
    #
    #     # 打印调试信息以确认参数是否正确
    #     print(f"Calling plot_single_phase with fig_num={fig_num}, labels={labels}")
    #
    #     # 确保传入正确的参数，包括 fig_num 和视角参数
    #     plot_single_phase(var_x, var_y, var_z, labels, title, fig_num, elev, azim)

    #testyz2()
