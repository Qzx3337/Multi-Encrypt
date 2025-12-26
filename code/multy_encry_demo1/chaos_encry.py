import gym
import textwrap
import cv2
import sys

# sys.path.append('/tmp/pycharm_project_60/code/gym-lorenz')  # 或者确切的安装路径（在服务器上运行时候）
import gym_lorenz
from PIL import Image

from stable_baselines3 import A2C
from stable_baselines3 import PPO

import random
import matplotlib.pyplot as plt
import numpy as np


''' 
GLOBAL Constants
'''
# Lorenz paramters and initial conditions
a, b, c = 10, 2.667, 28
x0, y0, z0 = 0, 0, 0

# M_image = 1024
# N_image = 1280

M_image = 512
N_image = 512

p = 8

# DNA-Encoding RULE #1 A = 00, T=01, G=10, C=11
dna = {}
dna["00"] = "A"
dna["01"] = "T"
dna["10"] = "G"
dna["11"] = "C"
dna["A"] = [0, 0]
dna["T"] = [0, 1]
dna["G"] = [1, 0]
dna["C"] = [1, 1]
# DNA xor
dna["AA"] = dna["TT"] = dna["GG"] = dna["CC"] = "A"
dna["AG"] = dna["GA"] = dna["TC"] = dna["CT"] = "G"
dna["AC"] = dna["CA"] = dna["GT"] = dna["TG"] = "C"
dna["AT"] = dna["TA"] = dna["CG"] = dna["GC"] = "T"
# Maximum time point and total number of time points
tmax, N = 100, 10000

coding_rules = {
    1: {'00': 'A', '11': 'T', '10': 'C', '01': 'G'},
    2: {'00': 'A', '11': 'T', '01': 'C', '10': 'G'},
    3: {'11': 'A', '00': 'T', '10': 'C', '01': 'G'},
    4: {'11': 'A', '00': 'T', '01': 'C', '10': 'G'},
    5: {'01': 'A', '10': 'T', '00': 'C', '11': 'G'},
    6: {'01': 'A', '10': 'T', '11': 'C', '00': 'G'},
    7: {'10': 'A', '01': 'T', '00': 'C', '11': 'G'},
    8: {'10': 'A', '01': 'T', '11': 'C', '00': 'G'}
}


class KalmanFilter1D:
    def __init__(self, initial_estimate, measurement_uncertainty, process_variance):
        self.estimate = initial_estimate
        self.estimate_error = measurement_uncertainty
        self.measurement_uncertainty = measurement_uncertainty
        self.process_variance = process_variance

    def update(self, measurement):
        # Prediction step
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # Update step
        kalman_gain = prediction_error / (prediction_error + self.measurement_uncertainty)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate


def get_original_img_path():
    path = "img/d01.png"
    return path


def get_encrypted_img_path():
    path = "img/encrypted_image.png"
    return path


def get_decrypted_img_path():
    path = "img/decrypted_image.png"
    return path



def split_into_rgb_channels(image):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    return red, green, blue


def decompose_matrix(iname):
    image = cv2.imread(iname)
    blue, green, red = split_into_rgb_channels(image)
    for values, channel in zip((red, green, blue), (2, 1, 0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype=np.uint8)
        img[:, :] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    return B, G, R


def logistic_map(theta, initial_value, num_iterations):
    """
    Generate a sequence using the Logistic map.

    Parameters:
    theta (float): Parameter controlling the behavior of the map.
    initial_value (float): Initial value of the sequence.
    num_iterations (int): Number of iterations to generate.

    Returns:
    list: A list containing the generated sequence.
    """
    # 确保 theta 和 initial_value 是标量
    sequence = [initial_value]

    for _ in range(num_iterations):
        last_value = sequence[-1]
        next_value = theta * last_value * (1 - last_value)
        sequence.append(next_value)

    return sequence


def mat_reshape(logistic_sequence):
    # 将列表转换为 NumPy 数组
    K1_array = np.array(logistic_sequence)

    # 应用变换
    K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)

    # 重塑为 (M, N) 形状的矩阵 Q
    Q = K1_prime.reshape(M_image, N_image)

    return Q


def split_into_blocks(matrix, p):
    blocks = []
    for i in range(0, M_image, p):  # 确保不超出边界
        for j in range(0, N_image, p):
            block = matrix[i:i + p, j:j + p]
            blocks.append(block)
    return blocks


def reshape_blocks(blocks, p):
    reshaped_matrix = np.zeros((M_image, N_image), dtype=np.uint8)
    index = 0
    for i in range(0, M_image, p):
        for j in range(0, N_image, p):
            reshaped_matrix[i:i + p, j:j + p] = blocks[index]
            index += 1
    return reshaped_matrix


def convert_to_8bit_binary(matrix_block_all):
    """
    将矩阵块中的每一个整数元素转换为8位的二进制字符串。

    参数:
        matrix_block_all (list of numpy.ndarray): 输入的矩阵块列表。

    返回:
        list of list: 每个元素都是8位二进制字符串的新矩阵块列表。
    """
    binary_block_all = []
    for matrix_block in matrix_block_all:
        # 使用np.vectorize将每个元素转换为8位二进制字符串
        binary_vectorizer = np.vectorize(lambda x: format(int(x), '08b') if str(x).isdigit() else '')
        binary_block = binary_vectorizer(matrix_block)

        # 将NumPy数组转换为普通列表
        binary_list = binary_block.tolist()
        binary_block_all.append(binary_list)

    return binary_block_all


def convert_binary_to_decimal(matrix_block_all):
    """
    将矩阵块中的每一个8位二进制字符串元素转换为0-255范围的整数。

    参数:
        matrix_block_all (list of list of str): 每个元素都是8位二进制字符串的输入矩阵块列表。

    返回:
        list of list: 每个元素都是0-255范围内整数的新矩阵块列表。
    """
    decimal_block_all = []

    for binary_block in matrix_block_all:
        decimal_block = []
        for row in binary_block:
            decimal_row = [int(binary_str, 2) if isinstance(binary_str, str) and len(binary_str) == 8 else 0 for
                           binary_str in row]
            decimal_block.append(decimal_row)
        decimal_block_all.append(decimal_block)

    return decimal_block_all


def binary_to_dna(binary_blocks, bd, coding_rules):
    """
    根据给定的二进制块、x1值和编码规则，将二进制字符串转换为DNA序列，并保持三维列表结构。

    参数:
        binary_blocks (list of list of list of str): 三维列表，每个元素是一个二进制字符串块。
        x1 (list of int): 每个块对应的x1值列表。
        coding_rules (dict): 包含不同x1值对应编码规则的字典。

    返回:
        list of list of list of str: 转换后的三维DNA序列列表。
    """
    dna_blocks = []

    # 确保 x1 和 binary_blocks 长度一致
    if len(bd) != len(binary_blocks):
        raise ValueError("The length of x1 and binary_blocks must be the same.")

    for i, block in enumerate(binary_blocks):
        # 选择当前块的编码规则
        rule = coding_rules.get(bd[i])
        if not rule:
            raise ValueError(f"No coding rule found for x1 value {bd[i]}.")

        # 初始化新的DNA编码块
        dna_block = []

        # 对每个二进制字符串应用编码规则
        for row in block:  # block 是一个二维列表
            dna_row = [''.join([rule[binary_str[j:j + 2]] for j in range(0, len(binary_str), 2)]) for binary_str in row]
            dna_block.append(dna_row)

        dna_blocks.append(dna_block)

    return dna_blocks


def dna_to_binary(dna_blocks, db, coding_rules):
    """
    根据给定的DNA序列、x2值和编码规则，将DNA序列转换为二进制字符串，并保持三维列表结构。

    参数:
        dna_blocks (list of list of list of str): 三维列表，每个元素是一个DNA序列块。
        x2 (list of int): 每个块对应的x2值列表。
        coding_rules (dict): 包含不同x2值对应编码规则的字典，规则应为两位二进制到单个DNA字符的映射。

    返回:
        list of list of list of str: 转换后的三维二进制字符串列表。
    """
    binary_blocks = []

    # 确保 x2 和 dna_blocks 长度一致
    if len(db) != len(dna_blocks):
        raise ValueError("The length of x2 and dna_blocks must be the same.")

    for i, block in enumerate(dna_blocks):
        # 选择当前块的编码规则，并创建其逆向映射
        rule = coding_rules.get(db[i])
        if not rule:
            raise ValueError(f"No coding rule found for x2 value {db[i]}.")

        # 创建逆向映射（DNA字符 -> 二进制）
        reverse_rule = {v: k for k, v in rule.items()}

        # 初始化新的二进制编码块
        binary_block = []

        # 对每个DNA字符串应用编码规则
        for row in block:  # block 是一个二维列表
            binary_row = []
            for dna_str in row:
                # 确保DNA字符串长度是4，以便能被分成两个字符的对
                if len(dna_str) != 4:
                    raise ValueError(f"DNA string {dna_str} is not exactly 4 characters long.")

                try:
                    # 将DNA字符串中的相邻字符转换成二进制字符串
                    binary_str = ''.join([reverse_rule[char] for char in dna_str])
                    binary_row.append(binary_str)
                except KeyError as e:
                    raise ValueError(f"Invalid DNA character {e} found in string {dna_str}.")
            binary_block.append(binary_row)

        binary_blocks.append(binary_block)

    return binary_blocks


def recover_image(b, g, r, iname, path):
    img = cv2.imread(iname)
    img[:, :, 2] = r
    img[:, :, 1] = g
    img[:, :, 0] = b
    cv2.imwrite((path), img)
    print("saved ecrypted image successfully")
    return img


def testdna(seq1, seq2):
    I1 = reshape_blocks(seq1, p)
    I2 = reshape_blocks(seq2, p)

    are_equivalent = np.array_equal(I1, I2)
    print("Are the matrices equivalent?", are_equivalent)  # 输出: Are the matrices equivalent? True
    print("I1 :")
    print(I1.flat[:10])  # 使用 .flat 获取一个迭代器，可以按行优先顺序访问元素

    print("\nI2 :")
    print(I2.flat[:10])


# 定义区间及其对应的目标整数
intervals = [
    (-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5)
]
targets = [1, 2, 3, 4, 5]  # 对应每个区间的整数

def map_to_integer(value):
    for (lower, upper), target in zip(intervals, targets):
        if lower <= value < upper:
            return target
    return round(value)  # 如果不在任何区间内，默认四舍五入


def process_array(arr):
    # 四舍五入并映射到区间
    # mapped_arr = np.array([map_to_integer(num) for num in arr])

    # 取模运算并转换为 uint8
    processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
    # processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
    return processed_arr


class _password:
    def __init__(self, dec_sequences=None):
        self.dec_sequences = dec_sequences


def encry2(x_arrays, original_img_path: str, encrypted_img_path: str):
    """
    加密过程
    """
    blue, green, red = decompose_matrix(original_img_path)  # 生成rgb M,N 1024, 1280

    i1 = np.sum(red)
    i2 = np.sum(green)
    theta = 3.9999  # Example parameter value
    initial_value = (i1 + i2) / (255 * M_image * N_image * 2)  # Example initial value
    num_iterations = M_image * N_image  # Number of iterations

    logistic_sequence = logistic_map(theta, initial_value, num_iterations - 1)
    # print(len(logistic_sequence))

    Q = mat_reshape(logistic_sequence)

    blocks_I1 = split_into_blocks(red, p)
    blocks_I2 = split_into_blocks(green, p)
    blocks_I3 = split_into_blocks(blue, p)

    test_I2_blocks = blocks_I2

    # 获取 Q 的子块
    blocks_Q = split_into_blocks(Q, p)

    # 应用 XOR 操作
    encrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
    encrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
    encrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

    bin_blocks_I1 = convert_to_8bit_binary(encrypted_blocks_I1)
    bin_blocks_I2 = convert_to_8bit_binary(encrypted_blocks_I2)
    bin_blocks_I3 = convert_to_8bit_binary(encrypted_blocks_I3)


    x1 = (np.mod(np.round(x_arrays[0]), 8) + 1).astype(np.uint8)
    x2 = (np.mod(np.round(x_arrays[1]), 8) + 1).astype(np.uint8)
    x3 = (np.mod(np.round(x_arrays[2]), 8) + 1).astype(np.uint8)
    x4 = (np.mod(np.round(x_arrays[3]), 8) + 1).astype(np.uint8)


    # 对每个数组应用该函数

    for i in range(15):
        x1 = process_array_with_kalman(x1)
        x2 = process_array_with_kalman(x2)
        x3 = process_array_with_kalman(x3)
        x4 = process_array_with_kalman(x4)

        x1 = process_array(x1)
        x2 = process_array(x2)
        x3 = process_array(x3)
        x4 = process_array(x4)


    # print(x1.flat[:20])
    # print(x5.flat[:20])

    # x5 = x1
    # x6 = x2
    # x7 = x3
    # x8 = x4

    # 调用函数进行转换
    dna_sequences_I1 = binary_to_dna(bin_blocks_I1, x1, coding_rules)

    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x2, coding_rules)

    dna_sequences_I2 = binary_to_dna(bin_blocks_I2, x1, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x2, coding_rules)

    dna_sequences_I3 = binary_to_dna(bin_blocks_I3, x1, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x2, coding_rules)

    dna_sequences_I1 = binary_to_dna(bin_sequences_I1, x3, coding_rules)
    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x4, coding_rules)

    dna_sequences_I2 = binary_to_dna(bin_sequences_I2, x3, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x4, coding_rules)

    dna_sequences_I3 = binary_to_dna(bin_sequences_I3, x3, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x4, coding_rules)

    dec_sequences_I1 = convert_binary_to_decimal(bin_sequences_I1)
    dec_sequences_I2 = convert_binary_to_decimal(bin_sequences_I2)
    dec_sequences_I3 = convert_binary_to_decimal(bin_sequences_I3)

    I1_prime = reshape_blocks(dec_sequences_I1, p)
    I2_prime = reshape_blocks(dec_sequences_I2, p)
    I3_prime = reshape_blocks(dec_sequences_I3, p)

    recover_image(I1_prime, I2_prime, I3_prime, original_img_path, encrypted_img_path)

    password = _password((dec_sequences_I1, dec_sequences_I2, dec_sequences_I3))
    return password



def extract_channels_from_path(path: str):
    """
    读取指定路径的图片，并分离出 R, G, B 通道数组。
    
    Args:
        path: 图片的路径
        
    Returns:
        r, g, b: 分别对应红、绿、蓝通道的 numpy 数组
    """
    # 1. 读取图片
    img = cv2.imread(path)
    
    # 检查图片是否读取成功
    if img is None:
        print(f"Error: 无法从路径 {path} 读取图片")
        return None, None, None

    # 2. 分离通道
    # 注意：OpenCV 读取的图片格式为 BGR (Blue, Green, Red)
    # img[:, :, 0] 是蓝色通道 (B)
    # img[:, :, 1] 是绿色通道 (G)
    # img[:, :, 2] 是红色通道 (R)
    
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    
    return b, g, r


def decry2(x_arrays, password: _password, encrypted_img_path: str, decrypted_img_path: str):
    """
    解密过程
    """

    original_img_path = get_original_img_path()

    blue, green, red = decompose_matrix(original_img_path)  # 生成rgb M,N 1024, 1280

    i1 = np.sum(red)
    i2 = np.sum(green)
    theta = 3.9999  # Example parameter value
    initial_value = (i1 + i2) / (255 * M_image * N_image * 2)  # Example initial value
    num_iterations = M_image * N_image  # Number of iterations

    logistic_sequence = logistic_map(theta, initial_value, num_iterations - 1)
    # print(len(logistic_sequence))

    Q = mat_reshape(logistic_sequence)

    blocks_Q = split_into_blocks(Q, p)

    
    x5 = (np.mod(np.round(x_arrays[4]), 8) + 1).astype(np.uint8)
    x6 = (np.mod(np.round(x_arrays[5]), 8) + 1).astype(np.uint8)
    x7 = (np.mod(np.round(x_arrays[6]), 8) + 1).astype(np.uint8)
    x8 = (np.mod(np.round(x_arrays[7]), 8) + 1).astype(np.uint8)

    for i in range(15):
        x5 = process_array_with_kalman(x5)
        x6 = process_array_with_kalman(x6)
        x7 = process_array_with_kalman(x7)
        x8 = process_array_with_kalman(x8)
        
        x5 = process_array(x5)
        x6 = process_array(x6)
        x7 = process_array(x7)
        x8 = process_array(x8)


    dec_sequences_I1 = password.dec_sequences[0]
    dec_sequences_I2 = password.dec_sequences[1]
    dec_sequences_I3 = password.dec_sequences[2]

    I1_prime = reshape_blocks(dec_sequences_I1, p)
    I2_prime = reshape_blocks(dec_sequences_I2, p)
    I3_prime = reshape_blocks(dec_sequences_I3, p)

    blocks_I1 = split_into_blocks(I1_prime, p)
    blocks_I2 = split_into_blocks(I2_prime, p)
    blocks_I3 = split_into_blocks(I3_prime, p)
    # print(len(blocks_I1))

    bin_blocks_I1 = convert_to_8bit_binary(dec_sequences_I1)
    bin_blocks_I2 = convert_to_8bit_binary(dec_sequences_I2)
    bin_blocks_I3 = convert_to_8bit_binary(dec_sequences_I3)

    # 调用函数进行转换
    dna_sequences_I1 = binary_to_dna(bin_blocks_I1, x8, coding_rules)
    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x7, coding_rules)

    dna_sequences_I2 = binary_to_dna(bin_blocks_I2, x8, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x7, coding_rules)

    dna_sequences_I3 = binary_to_dna(bin_blocks_I3, x8, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x7, coding_rules)


    # 调用函数进行转换
    dna_sequences_I1 = binary_to_dna(bin_sequences_I1, x6, coding_rules)
    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x5, coding_rules)

    dna_sequences_I2 = binary_to_dna(bin_sequences_I2, x6, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x5, coding_rules)

    dna_sequences_I3 = binary_to_dna(bin_sequences_I3, x6, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x5, coding_rules)

    dec_sequences_I1 = convert_binary_to_decimal(bin_sequences_I1)
    dec_sequences_I2 = convert_binary_to_decimal(bin_sequences_I2)
    dec_sequences_I3 = convert_binary_to_decimal(bin_sequences_I3)

    I1_reshape = reshape_blocks(dec_sequences_I1, p)
    I2_reshape = reshape_blocks(dec_sequences_I2, p)
    I3_reshape = reshape_blocks(dec_sequences_I3, p)

    # testdna(encrypted_blocks_I2, dec_sequences_I2)

    blocks_I1 = split_into_blocks(I1_reshape, p)
    blocks_I2 = split_into_blocks(I2_reshape, p)
    blocks_I3 = split_into_blocks(I3_reshape, p)

    # # 应用 XOR 操作
    dncrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
    dncrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
    dncrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

    I1_prime = reshape_blocks(dncrypted_blocks_I1, p)
    I2_prime = reshape_blocks(dncrypted_blocks_I2, p)
    I3_prime = reshape_blocks(dncrypted_blocks_I3, p)

    recover_image(I1_prime, I2_prime, I3_prime, get_original_img_path(), decrypted_img_path)

    # TODO: 将这些值作为返回值用于外部测试。
    # testdna(test_I2_blocks, dncrypted_blocks_I2)
    # are_equivalent = np.array_equal(green, I2_prime)
    # print("Are the matrices equivalent?", are_equivalent)  # 输出: Are the matrices equivalent? True


def test2(x_arrays):
    """
    TODO: 追踪替换 encrypath。
    """
    # print(x1.flat[:20])
    # print(x5.flat[:20])

    # ===== 加密 =====
    password = encry2(x_arrays, get_original_img_path(), get_encrypted_img_path())
    # ===== 加密完成 =====
    # ===== 解密 =====
    decry2(x_arrays, password, get_encrypted_img_path(), get_decrypted_img_path())
    # ===== 解密完成 =====



def generate(num):
    env = gym.make('lorenz_transient-v0')
    # model = PPO.load('../lorenz_gtp_m0', env, verbose=1)
    model = PPO.load('lorenz_targeting_1280k', env, verbose=1)
    # 创建并保存每种观测值对应的所有线条数据
    list_inital = []

    list_obs1 = []
    list_obs2 = []
    list_obs3 = []
    list_obs4 = []
    list_obs5 = []
    list_obs6 = []
    list_obs7 = []
    list_obs8 = []
    list_act1 = []

    obs = env.reset()

    for i in range(num):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        x1 = env.get_current()
        x2 = env.get_current1()
        x3 = env.get_current2()
        x4 = env.get_current3()
        if i > 1499:
            list_obs1.append(x1[0])
            list_obs2.append(x2[0])
            list_obs3.append(x3[0])
            list_obs4.append(x4[0])
            list_obs5.append(x1[1])
            list_obs6.append(x2[1])
            list_obs7.append(x3[1])
            list_obs8.append(x4[1])
            list_act1.append(action[0])

        if i == 0:
            list_inital.append([obs[0], obs[1], obs[2], obs[3], action[0]])

    return list_obs1, list_obs2, list_obs3, list_obs4, list_obs5, list_obs6, list_obs7, list_obs8


def calculate_correlation(image, direction):
    if direction == 'horizontal':
        return np.corrcoef(image[:, :-1].ravel(), image[:, 1:].ravel())[0, 1]
    elif direction == 'vertical':
        return np.corrcoef(image[:-1, :].ravel(), image[1:, :].ravel())[0, 1]
    elif direction == 'diagonal':
        h, w = image.shape[:2]
        diag = np.array([image[i, i] for i in range(min(h, w))])
        return np.corrcoef(diag[:-1], diag[1:])[0, 1]


def plot_correlation_distribution(image, channel, direction):
    if channel == 'B':
        channel_image = image[:, :, 0]
    elif channel == 'G':
        channel_image = image[:, :, 1]
    elif channel == 'R':
        channel_image = image[:, :, 2]

    if direction == 'horizontal':
        x = channel_image[:, :-1].ravel()
        y = channel_image[:, 1:].ravel()
    elif direction == 'vertical':
        x = channel_image[:-1, :].ravel()
        y = channel_image[1:, :].ravel()
    elif direction == 'diagonal':
        h, w = channel_image.shape
        diag = np.array([channel_image[i, i] for i in range(min(h, w))])
        x = diag[:-1]
        y = diag[1:]

    plt.scatter(x, y, s=1, alpha=0.5)
    plt.title(f'{direction.capitalize()} Direction Correlation Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Adjacent Pixel Value')
    plt.show()


def generate_dis_pic():
    """
    生成相关系数分布图
    """
    # 加密前图像路径
    original_image_path = get_original_img_path()
    # 加密后图像路径
    encrypted_image_path = get_encrypted_img_path()

    # 读取图像
    original_image = cv2.imread(original_image_path)
    encrypted_image = cv2.imread(encrypted_image_path)

    # 将BGR图像转换为RGB图像（因为OpenCV默认读取的是BGR格式）
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    encrypted_image_rgb = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

    # 计算相关系数
    correlations_original = {
        'horizontal': calculate_correlation(original_image_rgb, 'horizontal'),
        'vertical': calculate_correlation(original_image_rgb, 'vertical'),
        'diagonal': calculate_correlation(original_image_rgb, 'diagonal')
    }

    correlations_encrypted = {
        'horizontal': calculate_correlation(encrypted_image_rgb, 'horizontal'),
        'vertical': calculate_correlation(encrypted_image_rgb, 'vertical'),
        'diagonal': calculate_correlation(encrypted_image_rgb, 'diagonal')
    }

    # 打印相关系数
    print("Original Image Correlations:")
    for direction, correlation in correlations_original.items():
        print(f"{direction.capitalize()}: {correlation}")

    print("\nEncrypted Image Correlations:")
    for direction, correlation in correlations_encrypted.items():
        print(f"{direction.capitalize()}: {correlation}")

    # 绘制相关分布图
    plot_correlation_distribution(original_image_rgb, 'R', 'horizontal')
    plot_correlation_distribution(original_image_rgb, 'R', 'vertical')
    plot_correlation_distribution(original_image_rgb, 'R', 'diagonal')

    plot_correlation_distribution(encrypted_image_rgb, 'R', 'horizontal')
    plot_correlation_distribution(encrypted_image_rgb, 'R', 'vertical')
    plot_correlation_distribution(encrypted_image_rgb, 'R', 'diagonal')


def process_array_with_kalman(arr, measurement_uncertainty=1e-2, process_variance=1e-5):
    # 使用数组的第一个值作为初始估计值
    initial_estimate = arr[0]
    kf = KalmanFilter1D(initial_estimate, measurement_uncertainty, process_variance)

    # 对数组中的每个元素进行卡尔曼滤波
    filtered_arr = np.array([kf.update(num) for num in arr])

    print(filtered_arr.flat[:20])

    return filtered_arr


def load_image(path, channel='B'):
    img = Image.open(path)
    if channel == 'B':
        # 转换为灰度图像
        img = img.convert('L')
    else:
        # 提取特定颜色通道（未实现）
        r, g, b = img.split()
        if channel == 'R':
            img = r
        elif channel == 'G':
            img = g
        elif channel == 'B':
            img = b
    return np.array(img)


def calculate_correlation_distribution(img):
    h, w = img.shape

    horizontal_pairs = [(img[i, j], img[i, (j + 1) % w]) for i in range(h) for j in range(w - 1)]
    vertical_pairs = [(img[i, j], img[(i + 1) % h, j]) for i in range(h - 1) for j in range(w)]
    diagonal_pairs = [(img[i, j], img[i + 1, j + 1]) for i in range(h - 1) for j in range(w - 1) if
                      i < h - 1 and j < w - 1]

    return horizontal_pairs, vertical_pairs, diagonal_pairs


def calculate_correlation_coefficients(pairs_list):
    coefficients = []
    for pairs in pairs_list:
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]

        # 计算相关性系数
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_coefficient = correlation_matrix[0, 1]
        coefficients.append(correlation_coefficient)

    return coefficients



def process_and_visualize(image_path):
    """
    处理图像并可视化相关系数分布

    parameters: 图像路径

    读取图像，计算水平、垂直和对角线方向的相关系数分布，并绘制散点图。
    计算相关性系数并输出结果。
    生成相关系数分布图表。
    """

    def print_results(coefficients):
        print("Image\t\tHorizontal\tVertical\tDiagonal")
        print(f"Original Image\t{coefficients[0]:.5f}\t\t{coefficients[1]:.5f}\t\t{coefficients[2]:.5f}")


    def plot_correlation_distributions(pairs_list):
        fig, axs = plt.subplots(1, len(pairs_list), figsize=(15, 5))

        for ax, pairs in zip(axs, pairs_list):
            x_values = [pair[0] for pair in pairs]
            y_values = [pair[1] for pair in pairs]

            ax.scatter(x_values, y_values, s=1, c='blue', alpha=0.5)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)

        plt.tight_layout()
        plt.show()


    img = load_image(image_path, channel='B')
    horizontal_pairs, vertical_pairs, diagonal_pairs = calculate_correlation_distribution(img)
    pairs_list = [horizontal_pairs, vertical_pairs, diagonal_pairs]

    # 计算相关性系数
    coefficients = calculate_correlation_coefficients(pairs_list)

    # 输出结果
    print_results(coefficients)

    # 绘制图表
    plot_correlation_distributions(pairs_list)


import numpy as np
from PIL import Image

def calculate_entropy(image_path):
    """
    计算图像的信息熵（范围为0-8）

    parameters: 图像路径

    信息熵是衡量图像信息量和复杂度的指标，反映了图像中像素值分布的均匀程度。

    信息熵越高，图像越混乱随机。
    """
    # 打开图像并转换为灰度图像
    img = Image.open(image_path).convert('L')
    # 转换为numpy数组
    img_array = np.array(img)

    # 计算直方图
    histogram, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # 将直方图转换为概率分布
    probabilities = histogram / float(np.sum(histogram))

    # 过滤掉零概率，避免log(0)错误
    probabilities = probabilities[probabilities > 0]

    # 计算信息熵
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy



def NPCR(img1, img2):
    '''
    计算像素数变化率 (NPCR)

    parameters: 图像路径

    NCPR，图像加密指标，衡量两幅图像之间发生变化的像素的比例，用于评估加密算法的扩散性及抗差分攻击能力。
    
    NPCR 值越高，表示两幅图像之间的差异越大。
    '''
    # opencv颜色通道顺序为BGR
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape

    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((R1 != R2), return_counts=True)
    R_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)
    ar, num = np.unique((G1 != G2), return_counts=True)
    G_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)
    ar, num = np.unique((B1 != B2), return_counts=True)
    B_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)

    return R_npcr, G_npcr, B_npcr



def UACI(img1, img2):
    """
    计算两张图像之间的平均变化强度 (UACI)

    衡量两幅图像之间像素值的平均变化幅度，用于评估加密图像与原图（或另一加密图）的差异强度。

    UACI 值越高，表示两幅图像之间的差异越大。

    parameters: 图像路径
    """

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 元素为uint8类型取值范围：0到255
    # print(R1.dtype)

    # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)

    sumR = np.sum(abs(R1 - R2))
    sumG = np.sum(abs(G1 - G2))
    sumB = np.sum(abs(B1 - B2))
    R_uaci = sumR / 255 / (w * h)
    G_uaci = sumG / 255 / (w * h)
    B_uaci = sumB / 255 / (w * h)

    return R_uaci, G_uaci, B_uaci



# program exec9
if __name__ == "__main__":


    num = int((M_image * N_image) / (p * p))

    # list_x1,list_x2,list_x3,list_x4,list_x5,list_x6,list_x7,list_x8 = generate(num+1500)

    # x1 = np.array(list_x1)
    # x2 = np.array(list_x2)
    # x3 = np.array(list_x3)
    # x4 = np.array(list_x4)
    # x5 = np.array(list_x5)
    # x6 = np.array(list_x6)
    # x7 = np.array(list_x7)
    # x8 = np.array(list_x8)


    x_lists = generate(num + 1500)
    x_arrays = [np.array(lst) for lst in x_lists]

    #test(encrypath, x1, x2, x3, x4,x5,x6,x7,x8)
    test2(x_arrays)


    # generate_dis_pic()

    # image_path = get_encrypted_img_path()
    # entropy = calculate_entropy(image_path)
    # print(entropy)


    # image_paths = [
    #     'D:\\pythonmain\\opencv\\encryanddecry\\code\\chaos_apl\\LenaRGB.bmp',
    #     'D:\\pythonmain\\opencv\\encryanddecry\\code\\chaos_apl\\test.png'
    # ]
    
    # for path in image_paths:
    #     process_and_visualize(path)

    # # 示例使用 PIL 加载图像并转换为灰度图像
    # from PIL import Image
    #
    # img1_path = 'path_to_image1.png'
    # img2_path = 'path_to_image2.png'
    #
    # img1 = np.array(Image.open(img1_path).convert('L'))
    # img2 = np.array(Image.open(img2_path).convert('L'))
    #
    # R_npcr, G_npcr, B_npcr = NPCR(img1, img2)
    # print('*********PSNR*********')
    # # 百分数表示，保留小数点后4位
    # print('Red  :{:.4%}'.format(R_npcr))
    # print('Green:{:.4%}'.format(G_npcr))
    # print('Blue :{:.4%}'.format(B_npcr))
    #
    # R_uaci, G_uaci, B_uaci = UACI(img1, img2)
    # print('*********UACI*********')
    # # 百分数表示，保留小数点后4位
    # print('Red  :{:.4%}'.format(R_uaci))
    # print('Green:{:.4%}'.format(G_uaci))
    # print('Blue :{:.4%}'.format(B_uaci))
