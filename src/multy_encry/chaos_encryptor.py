import gym
import textwrap
import cv2
import sys

# sys.path.append('/tmp/pycharm_project_60/code/gym-lorenz') 
# sys.path.append('/code/multy_encry_demo1/')  

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


def get_plain_img_path():
    path = "data1/image_s0002_i0001.jpg"
    return path


def get_cipher_img_path():
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


def decompose_matrix(image: np.ndarray):
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


# def recover_image(b, g, r, img: np.ndarray, path):
#     img[:, :, 2] = r
#     img[:, :, 1] = g
#     img[:, :, 0] = b
#     cv2.imwrite((path), img)
#     print("image saved to:", path)
#     return img


def recover_image(b, g, r, path):
    """
    使用全局定义的尺寸 (M_image, N_image) 重组 BGR 通道并保存图像。
    不再依赖外部传入的 img 对象，消除了副作用。
    """
    # 1. 创建全新的空白画布
    # 注意：OpenCV 图片是 (Height, Width, Channels) -> (M, N, 3)
    # dtype=np.uint8 是必须的，确保像素值在 0-255 之间
    img = np.zeros((M_image, N_image, 3), dtype=np.uint8)

    # 2. 填充通道 (OpenCV 默认顺序为 B-G-R)
    img[:, :, 0] = b  # Blue Channel
    img[:, :, 1] = g  # Green Channel
    img[:, :, 2] = r  # Red Channel

    # 3. 保存并返回
    cv2.imwrite(path, img)
    print("image saved to:", path)
    
    return img

def testdna(seq1, seq2):
    I1 = reshape_blocks(seq1, p)
    I2 = reshape_blocks(seq2, p)

    are_equivalent = np.array_equal(I1, I2)
    print("Are the matrices equivalent?", are_equivalent)  # 输出: Are the matrices equivalent? True
    # print("I1 :")
    # print(I1.flat[:10])  # 使用 .flat 获取一个迭代器，可以按行优先顺序访问元素

    # print("\nI2 :")
    # print(I2.flat[:10])


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


def encrypt(seed: tuple, plain_path: str, cipher_path: str)-> tuple:
    x1, x2, x3, x4, x5, x6, x7, x8 = seed
    # print(x1.flat[:20])
    # print(x5.flat[:20])
    plain_img = cv2.imread(plain_path)
    if plain_img is None:
        raise FileNotFoundError(f"Unable to load image at {plain_path}")
    
    blue, green, red = decompose_matrix(plain_img)  # 生成rgb M,N 1024, 1280


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

    # 获取 Q 的子块
    blocks_Q = split_into_blocks(Q, p)

    # 应用 XOR 操作
    encrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
    encrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
    encrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

    bin_blocks_I1 = convert_to_8bit_binary(encrypted_blocks_I1)
    bin_blocks_I2 = convert_to_8bit_binary(encrypted_blocks_I2)
    bin_blocks_I3 = convert_to_8bit_binary(encrypted_blocks_I3)


    x1 = (np.mod(np.round(x1), 8) + 1).astype(np.uint8)
    x2 = (np.mod(np.round(x2) , 8) + 1).astype(np.uint8)
    x5 = (np.mod(np.round(x5), 8) + 1).astype(np.uint8)
    x6 = (np.mod(np.round(x6), 8) + 1).astype(np.uint8)
    x3 = (np.mod(np.round(x3) , 8) + 1).astype(np.uint8)
    x4 = (np.mod(np.round(x4) , 8) + 1).astype(np.uint8)
    x7 = (np.mod(np.round(x7), 8) + 1).astype(np.uint8)
    x8 = (np.mod(np.round(x8), 8) + 1).astype(np.uint8)


    def process_array(arr):
        # 四舍五入并映射到区间
        # mapped_arr = np.array([map_to_integer(num) for num in arr])

        # 取模运算并转换为 uint8
        processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
        # processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
        return processed_arr

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

        x5 = process_array_with_kalman(x5)
        x6 = process_array_with_kalman(x6)
        x7 = process_array_with_kalman(x7)
        x8 = process_array_with_kalman(x8)

        x5 = process_array(x5)
        x6 = process_array(x6)
        x7 = process_array(x7)
        x8 = process_array(x8)


    # print(x1.flat[:20])
    # print(x5.flat[:20])


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

    # recover_image(I1_prime, I2_prime, I3_prime, plain_img, cipher_path)
    recover_image(I1_prime, I2_prime, I3_prime, cipher_path)

    dec_seed = (x5, x6, x7, x8)
    dec_sequences = (dec_sequences_I1, dec_sequences_I2, dec_sequences_I3)
    decry_key = (dec_seed, dec_sequences, blocks_Q)
    return decry_key


def decrypt(decry_key: tuple, cipher_path: str, decrypted_path: str):

    dec_seed, dec_sequences, blocks_Q = decry_key
    x5, x6, x7, x8 = dec_seed
    dec_sequences_I1, dec_sequences_I2, dec_sequences_I3 = dec_sequences
    I1_prime = reshape_blocks(dec_sequences_I1, p)
    I2_prime = reshape_blocks(dec_sequences_I2, p)
    I3_prime = reshape_blocks(dec_sequences_I3, p)
    
    cipher_img = cv2.imread(cipher_path)
    if cipher_img is None:
        print(f"Error: Unable to load image at {cipher_path}")
        return False


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
    
    # recover_image(I1_prime, I2_prime, I3_prime, cipher_img, decrypted_path)
    recover_image(I1_prime, I2_prime, I3_prime, decrypted_path)

    # ... code ...



def test2(seed, plain_path=get_plain_img_path(), cipher_path=get_cipher_img_path(), decrypted_path=get_decrypted_img_path()):

    decry_key = encrypt(seed, plain_path, cipher_path)

    decrypt(decry_key, cipher_path, decrypted_path)
    # =========================================================

    # testdna(test_I2_blocks, dncrypted_blocks_I2)

    # are_equivalent = np.array_equal(green, I2_prime)
    # print("Are the matrices equivalent?", are_equivalent)  # 输出: Are the matrices equivalent? True



def generate(num):
    env = gym.make('lorenz_transient-v0')
    model = PPO.load('experiments/exp_lorenz/lorenz_f2_lr5en5_s1m.zip', env, verbose=1)
    # model = PPO.load('lorenz_targeting_810k', env, verbose=1)
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


def plot_rgb_histogram(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否成功加载
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # 将BGR图像转换为RGB图像（因为OpenCV默认读取的是BGR格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算每个通道的直方图
    hist_red, bins_red = np.histogram(image_rgb[:, :, 0].ravel(), bins=256, range=[0, 256])
    hist_green, bins_green = np.histogram(image_rgb[:, :, 1].ravel(), bins=256, range=[0, 256])
    hist_blue, bins_blue = np.histogram(image_rgb[:, :, 2].ravel(), bins=256, range=[0, 256])

    # 定义绘制直方图的函数
    def plot_histogram(hist, bins, color):
        plt.figure(figsize=(6, 4))
        plt.bar(bins[:-1], hist, width=1, color=color, alpha=0.7)
        plt.xlim([0, 256])
        plt.show()

    # 绘制并显示红色通道的直方图
    plot_histogram(hist_red, bins_red, 'red')

    # 绘制并显示绿色通道的直方图
    plot_histogram(hist_green, bins_green, 'green')

    # 绘制并显示蓝色通道的直方图
    plot_histogram(hist_blue, bins_blue, 'blue')


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


def generate_dis_pic(plain_path=get_plain_img_path(), cipher_path=get_cipher_img_path()):
    """
    生成原图和加密图的相关系数及相关分布图
    """
    # 加密前图像路径
    original_image_path = plain_path
    # 加密后图像路径
    encrypted_image_path = cipher_path

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

    # print(filtered_arr.flat[:20])

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


def process_and_visualize(image_path):
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


def UACI(img1_path: str, img2_path: str):
    """    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
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


def generate_seed():
    num = int((M_image * N_image) / (p * p))
    list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, list_x8 = generate(num + 1500)

    x1 = np.array(list_x1)
    x2 = np.array(list_x2)
    x3 = np.array(list_x3)
    x4 = np.array(list_x4)
    x5 = np.array(list_x5)
    x6 = np.array(list_x6)
    x7 = np.array(list_x7)
    x8 = np.array(list_x8)

    return x1, x2, x3, x4, x5, x6, x7, x8


def check_decryption_pixel(plain_path, decrypted_path):
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    if plain_img.shape != decrypted_img.shape:
        return False
    difference = cv2.absdiff(plain_img, decrypted_img)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    else:
        return False


def check_decryption_psnr(plain_path: str, decrypted_path: str):
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    
    if plain_img.shape != decrypted_img.shape:
        return False

    # 计算 PSNR
    psnr_value = cv2.PSNR(plain_img, decrypted_img)
    
    # 设定阈值，比如 40dB，意味着差异极小
    if psnr_value > 40: 
        return True
    else:
        return False


def encrypt_and_decrypt_once(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None):
    if plain_path is None or cipher_path is None or decrypted_path is None:
        raise ValueError("File paths cannot be None.")
    seed = generate_seed()
    decry_key = encrypt(seed, plain_path, cipher_path)
    decrypt(decry_key, cipher_path, decrypted_path)
    if check_decryption_psnr(plain_path, decrypted_path):
        print("Decryption successful: The decrypted image matches the original.")
        if not check_decryption_pixel(plain_path, decrypted_path):
            print("Warning: Decryption verification failed: Pixel values do not match exactly.")
        else:
            print("Verification succeed: Pixel values match exactly.")
        return True
    else:
        print("Error: Decryption failed. The decrypted image does not match the original.")
        return False


def encrypt_and_decrypt(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None):
    cnt = 0
    while True:
        cnt += 1
        print("********** Round: {} **********".format(cnt))
        flag = encrypt_and_decrypt_once(plain_path, cipher_path, decrypted_path)
        if flag:
            break
        elif cnt >= 10:
            raise Exception("Decryption failed after 10 attempts.")
            

import os
def process_images_in_folder(source_dir, cipher_dir, decrypted_dir):
    """
    遍历 source_dir 下的所有 png 图片，进行加密和解密，
    并将结果分别保存到 cipher_dir 和 decrypted_dir。
    如果 decrypted_dir 中已存在同名文件，则跳过（增量更新）。
    """
    # 1. 确保输出目录存在，如果不存在则创建
    os.makedirs(cipher_dir, exist_ok=True)
    os.makedirs(decrypted_dir, exist_ok=True)

    # 2. 获取源目录下的所有文件
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    files = os.listdir(source_dir)
    count_processed = 0
    count_skipped = 0

    print(f"Starting batch processing in: {source_dir}\n" + "-"*40)

    for file_name in files:
        # 3. 过滤文件，只处理 png (忽略大小写)
        if file_name.lower().endswith('.png'):
            
            # 构造完整路径
            plain_path = os.path.join(source_dir, file_name)
            cipher_path = os.path.join(cipher_dir, file_name)
            decrypted_path = os.path.join(decrypted_dir, file_name)

            # 4. 核心逻辑：检查是否已经处理过
            # 如果解密文件夹里已经有了这个图，说明之前跑过，直接跳过
            print()
            if os.path.exists(decrypted_path):
                print(f"[SKIP] {file_name} already exists in decrypted folder.")
                count_skipped += 1
                continue

            # 5. 开始处理新图片
            print(f"[ACTION] Processing new image: {file_name}")
            try:
                # 调用你写好的带有重试机制的函数
                encrypt_and_decrypt(plain_path, cipher_path, decrypted_path)
                count_processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {file_name}. Reason: {str(e)}")
    
    print("-" * 40)
    print(f"Batch task finished. New Processed: {count_processed}, Skipped: {count_skipped}.")
    pass


if __name__ == "__main__":
    # 定义文件夹路径
    plain_folder = "data/test_img/plain_img"          # 未加密图片文件夹
    cipher_folder = "data/test_img/cipher_img"        # 加密后存放文件夹
    decrypted_folder = "data/test_img/decrypted_img"  # 解密后存放文件夹

    # 确保源文件夹存在，否则无法处理
    if os.path.exists(plain_folder):
        try:
            process_images_in_folder(plain_folder, cipher_folder, decrypted_folder)
        except Exception as e:
            print("An error occurred during batch processing:", str(e))
    else:
        print(f"Error: Source directory '{plain_folder}' not found.")


# if __name__ == "__main__":

    # test2(generate_seed())

    # cnt = 0
    # while True:
    #     cnt += 1
    #     print("\n********** Test Round {} **********".format(cnt))
    #     flag = encrypt_and_decrypt(get_plain_img_path(), get_cipher_img_path(), get_decrypted_img_path())
    #     if flag:
    #         break

    # try:
    #     encrypt_and_decrypt(get_plain_img_path(), get_cipher_img_path(), get_decrypted_img_path())
    # except Exception as e:
    #     print("An error occurred during encryption/decryption:", str(e))

    # pass

    # plot_rgb_histogram(get_plain_img_path())
    # plot_rgb_histogram(get_cipher_img_path())


    # generate_dis_pic()

 
    # entropy = calculate_entropy(get_cipher_img_path())
    # print(entropy)
 

    # image_paths = [
    #     get_plain_img_path(),
    #     get_cipher_img_path()
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
