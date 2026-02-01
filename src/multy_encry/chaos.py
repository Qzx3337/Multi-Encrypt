
import os
import traceback
import gym
import gym_lorenz
from stable_baselines3 import PPO
# from stable_baselines3 import A2C
import numpy as np
import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
import sys
import hashlib
# import random
# import textwrap


# --- 配置路径 (集中管理) ---

# 定义实验的主根目录 
BASE_EXPERIMENT_DIR = "experiments/w502"
# 定义数据实验目录
DATA_EXPERIMENT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "hyper_kvasir")

# 定义输入目录 
PLAIN_DIR = os.path.join(DATA_EXPERIMENT_DIR, "plain_img")   # 原图文件夹
CIPHER_DIR = os.path.join(DATA_EXPERIMENT_DIR, "cipher_img") # 密图文件夹
DECRYPTED_DIR = os.path.join(DATA_EXPERIMENT_DIR, "decrypted_img") # 解密图文件夹

# 模型文件所在目录
MODEL_DIR = "experiments/exp_lorenz/lorenz_f2_lr5en5_s1m.zip"

# --- ----------------- ---

''' 
GLOBAL Constants
'''
# Lorenz paramters and initial conditions
# a, b, c = 10, 2.667, 28
# x0, y0, z0 = 0, 0, 0


M_image = 512
N_image = 512
p = 8
MY_PASSWORD = "password987"  # 你可以修改为任何你想要的密码甚至为None

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


def string_to_initial_value(password: str) -> float:
    """
    将用户输入的字符串密码转换为 (0, 1) 之间的浮点数，用作混沌系统的初值。
    """
    # 1. 使用 SHA-256 计算哈希 (得到一个 256位的 16进制字符串)
    hash_obj = hashlib.sha256(password.encode('utf-8'))
    hex_dig = hash_obj.hexdigest()

    # 2. 将 16进制 转为巨大的整数
    int_val = int(hex_dig, 16)

    # 3. 归一化：除以 2^256 (SHA-256的最大可能值)，得到 0.0 ~ 1.0 之间的小数
    float_val = int_val / (2 ** 256)
    
    # 避免正好是 0 或 1 (虽然概率极低，但混沌系统对边缘值敏感)
    if float_val == 0: float_val = 0.123456789
    if float_val == 1: float_val = 0.987654321
    
    return float_val


def split_channels(image: np.ndarray):
    """
    拆分图像的三个通道BGR
    
    Args:
        image (np.ndarray): 输入的图像矩阵，形状为 (M, N, C)。
    Returns:
        (B, G, R) ((np.ndarray, np.ndarray, np.ndarray)): 包含三个 W x H 矩阵的元组，分别对应 B、G、R 通道。
    """
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
 
    return blue, green, red


def logistic_map(theta:float, initial_value:float, num_iterations:int) -> list:
    """
    Generate a sequence using the Logistic map.

    Args:
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


def reshape_sequence_to_Q(logistic_sequence: list) -> np.ndarray:
    """
    将混沌序列转换为掩码矩阵 Q。
    Args:
        logistic_sequence (list of float): 混沌序列列表，其中的变量为浮点数。
    Returns:
        Q (np.ndarray): 掩码矩阵, 形状为 (M, N).
    """
    # 将列表转换为 NumPy 数组
    K1_array = np.array(logistic_sequence)

    # 量化：(0, 1)浮点数转换为 0-255 范围内的整数
    K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)

    # 一维向量重塑为 (M, N) 形状的矩阵 Q
    Q = K1_prime.reshape(M_image, N_image)

    return Q


def split_into_blocks(matrix: np.ndarray, p: int)-> list:
    """
    将二维矩阵拆分为一系列 p x p 的小块。
    
    Args:
        matrix (np.ndarray): 图像的某个通道(W x H)
        p (int): 每个正方形小块的变长
    Returns:
        blocks (list of np.ndarray): 包含所有 p x p 小块的列表.
    """
    blocks = []
    for i in range(0, M_image, p):  # 确保不超出边界
        for j in range(0, N_image, p):
            block = matrix[i:i + p, j:j + p]
            blocks.append(block)
    return blocks


def reshape_blocks_to_channel(blocks, p):
    reshaped_matrix = np.zeros((M_image, N_image), dtype=np.uint8)
    index = 0
    for i in range(0, M_image, p):
        for j in range(0, N_image, p):
            reshaped_matrix[i:i + p, j:j + p] = blocks[index]
            index += 1
    return reshaped_matrix


def convert_to_8bit_binary(matrix_block_all: list) -> list:
    """
    将矩阵块中的每一个整数元素转换为8位的二进制字符串。

    Args:
        matrix_block_all (list of ny.ndarray, p x p x len): 输入的矩阵块列表。

    Returns:
        binary_block_all (list of list, p x p x len): 每个元素都是8位二进制字符串的新矩阵块列表。

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

    Args:
        matrix_block_all (list of list of str): 每个元素都是8位二进制字符串的输入矩阵块列表。

    Returns:
        decimal_block_all (list of list): 每个元素都是0-255范围内整数的新矩阵块列表。
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

    Args:
        binary_blocks (list of list of list of str): 三维列表，每个元素是一个二进制字符串块。
        x1 (list of int): 每个块对应的x1值列表。
        coding_rules (dict): 包含不同x1值对应编码规则的字典。

    Returns:
        (list of list of list of str): 转换后的三维DNA序列列表。
    """
    dna_blocks = []

    # 确保 lorenz sequence, x1 和 binary_blocks 长度一致
    if len(bd) != len(binary_blocks):
        print(f"len(x1): {len(bd)}, len(binary_blocks): {len(binary_blocks)}")
        raise ValueError("The length of lorenz sequence x1 and binary_blocks must be the same.")

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

    Args:
        dna_blocks (list of list of list of str): 三维列表，每个元素是一个DNA序列块。
        x2 (list of int): 每个块对应的x2值列表。
        coding_rules (dict): 包含不同x2值对应编码规则的字典，规则应为两位二进制到单个DNA字符的映射。

    Returns:
        binary_blocks (list of list of list of str): 转换后的三维二进制字符串列表。
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

def save_image(multi_channel_img: np.ndarray, path: str):
    """
    应对通道数可能为1或3或4的多通道图像，保存为tiff格式。
    Args:
        multi_channel_img (np.ndarray): 多通道图像矩阵，形状为 (M, N) 或 (M, N, C)。
                                        默认输入为 uint8 类型，且为 BGR 顺序。
        path (str): 保存图像的文件路径（建议以 .tif 或 .tiff 结尾）。
    
    Raises:
        ValueError: 当图像通道数不是 1, 3, 或 4 时抛出。
        IOError: 当图像保存失败时抛出。
    """
    
    # 1. 检查数据维度并确定通道数
    ndim = multi_channel_img.ndim
    if ndim == 2:
        # 形状为 (M, N)，视为单通道灰度图
        channels = 1
    elif ndim == 3:
        # 形状为 (M, N, C)
        channels = multi_channel_img.shape[2]
    else:
        raise ValueError(f"不支持的图像维度: {ndim}。图像应为 2D 或 3D 矩阵。")

    # 2. 检查通道数是否符合要求 (1, 3, 4)
    if channels not in [1, 3, 4]:
        raise ValueError(f"不支持的通道数: {channels}。仅支持 1 (Gray), 3 (BGR), 或 4 (BGRA) 通道。")

    # 3. 确保目标目录存在
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 4. 保存图像
    # cv2.imwrite 默认处理:
    # - 1通道: 保存为灰度
    # - 3通道: 默认为 BGR 顺序保存
    # - 4通道: 默认为 BGRA 顺序保存
    success = cv2.imwrite(path, multi_channel_img)

    if not success:
        raise IOError(f"图像保存失败，请检查路径是否有效或权限问题: {path}")

    print(f"成功保存 {channels} 通道图像至: {path}")


def process_array_with_kalman(arr, measurement_uncertainty=1e-2, process_variance=1e-5):
    # 使用数组的第一个值作为初始估计值
    initial_estimate = arr[0]
    kf = KalmanFilter1D(initial_estimate, measurement_uncertainty, process_variance)

    # 对数组中的每个元素进行卡尔曼滤波
    filtered_arr = np.array([kf.update(num) for num in arr])

    # print(filtered_arr.flat[:20])

    return filtered_arr


def encrypt(
        master_sequence: tuple, 
        plain_img: np.ndarray, 
        password: str = None
    ) -> tuple:
    x1, x2, x3, x4 = master_sequence
    # print(x1.flat[:20])
    # print(x5.flat[:20])

    # ===== 阶段1：Q置乱 =====

    # ===== 阶段1.1：明文的通道拆分=====

    blue, green, red = split_channels(plain_img)

    # ===== 阶段1.2：生成掩码矩阵Q =====

    # 原本的初始值生成方式：基于明文信息提取
    i1 = np.sum(red)
    i2 = np.sum(green)
    i3 = np.sum(blue)
    # initial_value = (i1 + i2) / (255 * M_image * N_image * 2) 

    # 带密码的初始值生成方式：图像信息与用户密码混合
    img_word = str(int(i1) + int(i2) + int(i3))

    # 检查password必须是字符串
    if isinstance(password, str):
        raw_seed_str = password + img_word
    elif password is None:
        raw_seed_str = img_word
    else:
        raise ValueError("Password must be a string or None.")
    
    initial_value = string_to_initial_value(raw_seed_str)
    theta = 3.9999  # Example parameter value
    num_iterations = M_image * N_image  # Number of iterations

    logistic_sequence = logistic_map(theta, initial_value, num_iterations - 1)
    # print(len(logistic_sequence))

    Q = reshape_sequence_to_Q(logistic_sequence)

    # ===== 阶段1.3：通道分别与Q做异或 =====

    blocks_I1 = split_into_blocks(blue, p)
    blocks_I2 = split_into_blocks(green, p)
    blocks_I3 = split_into_blocks(red, p)

    blocks_Q = split_into_blocks(Q, p)

    encrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
    encrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
    encrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

    # ===== 阶段2：DNA加密 =====

    # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则 =====

    x1 = (np.mod(np.round(x1), 8) + 1).astype(np.uint8)
    x2 = (np.mod(np.round(x2), 8) + 1).astype(np.uint8)
    x3 = (np.mod(np.round(x3), 8) + 1).astype(np.uint8)
    x4 = (np.mod(np.round(x4), 8) + 1).astype(np.uint8)

    def process_array(arr):
        processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
        return processed_arr

    # 滤波与量化掺杂进行
    for i in range(15):
        x1 = process_array_with_kalman(x1)
        x2 = process_array_with_kalman(x2)
        x3 = process_array_with_kalman(x3)
        x4 = process_array_with_kalman(x4)

        x1 = process_array(x1)
        x2 = process_array(x2)
        x3 = process_array(x3)
        x4 = process_array(x4)

    # ===== 阶段2.2：各通道转换为8位二进制字符串 =====
    bin_blocks_I1 = convert_to_8bit_binary(encrypted_blocks_I1)
    bin_blocks_I2 = convert_to_8bit_binary(encrypted_blocks_I2)
    bin_blocks_I3 = convert_to_8bit_binary(encrypted_blocks_I3)

    # ===== 阶段2.3：基于DNA编解码的加密 =====
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

    # ===== 阶段3：保存图像与生成密钥 =====

    dec_sequences_I1 = convert_binary_to_decimal(bin_sequences_I1)
    dec_sequences_I2 = convert_binary_to_decimal(bin_sequences_I2)
    dec_sequences_I3 = convert_binary_to_decimal(bin_sequences_I3)

    I1_prime = reshape_blocks_to_channel(dec_sequences_I1, p)
    I2_prime = reshape_blocks_to_channel(dec_sequences_I2, p)
    I3_prime = reshape_blocks_to_channel(dec_sequences_I3, p)

    cipher_img = cv2.merge((I1_prime, I2_prime, I3_prime))
    decry_key = (img_word, initial_value)

    return cipher_img, decry_key


def decrypt(
        slave_sequence: tuple,
        cipher_img: np.ndarray, 
        decry_key: tuple, 
        password: str = None
    ) -> tuple:

    # 对密钥解包
    img_word, initial_value_of_Q = decry_key

    # 验证用户是否输入了正确的密码
    if isinstance(password, str):
        raw_seed_str = password + img_word
    elif password is None:
        raw_seed_str = img_word
    else:
        raise ValueError("Password must be a string or None.")
    expected_initial_value = string_to_initial_value(raw_seed_str)
    if not np.isclose(expected_initial_value, initial_value_of_Q):
        print("Error: Incorrect password provided for decryption.")
        return (False, None)
    
    # 重新生成 Q 矩阵
    theta = 3.9999
    num_iterations = M_image * N_image  # Number of iterations
    logistic_sequence = logistic_map(theta, initial_value_of_Q, num_iterations - 1)
    Q = reshape_sequence_to_Q(logistic_sequence)
    blocks_Q = split_into_blocks(Q, p)

    # 量化混沌序列
    x5, x6, x7, x8 = slave_sequence
    
    x5 = (np.mod(np.round(x5), 8) + 1).astype(np.uint8)
    x6 = (np.mod(np.round(x6), 8) + 1).astype(np.uint8)
    x7 = (np.mod(np.round(x7), 8) + 1).astype(np.uint8)
    x8 = (np.mod(np.round(x8), 8) + 1).astype(np.uint8)

    def process_array(arr):
        processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
        return processed_arr

    for i in range(15):
        x5 = process_array_with_kalman(x5)
        x6 = process_array_with_kalman(x6)
        x7 = process_array_with_kalman(x7)
        x8 = process_array_with_kalman(x8)

        x5 = process_array(x5)
        x6 = process_array(x6)
        x7 = process_array(x7)
        x8 = process_array(x8)

    # 密文通道拆分
    blue_c, green_c, red_c = split_channels(cipher_img)

    # 分块
    blocks_I1 = split_into_blocks(blue_c, p)
    blocks_I2 = split_into_blocks(green_c, p)
    blocks_I3 = split_into_blocks(red_c, p) 

    # DNA解密

    # 转换为8位二进制字符串
    bin_blocks_I1 = convert_to_8bit_binary(blocks_I1)
    bin_blocks_I2 = convert_to_8bit_binary(blocks_I2)
    bin_blocks_I3 = convert_to_8bit_binary(blocks_I3)

    # 基于DNA编解码的解密
    dna_sequences_I1 = binary_to_dna(bin_blocks_I1, x8, coding_rules)
    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x7, coding_rules)
    dna_sequences_I2 = binary_to_dna(bin_blocks_I2, x8, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x7, coding_rules)
    dna_sequences_I3 = binary_to_dna(bin_blocks_I3, x8, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x7, coding_rules)

    dna_sequences_I1 = binary_to_dna(bin_sequences_I1, x6, coding_rules)
    bin_sequences_I1 = dna_to_binary(dna_sequences_I1, x5, coding_rules)
    dna_sequences_I2 = binary_to_dna(bin_sequences_I2, x6, coding_rules)
    bin_sequences_I2 = dna_to_binary(dna_sequences_I2, x5, coding_rules)
    dna_sequences_I3 = binary_to_dna(bin_sequences_I3, x6, coding_rules)
    bin_sequences_I3 = dna_to_binary(dna_sequences_I3, x5, coding_rules)

    # 转换回十进制整数
    blocks_I1 = convert_binary_to_decimal(bin_sequences_I1)
    blocks_I2 = convert_binary_to_decimal(bin_sequences_I2)
    blocks_I3 = convert_binary_to_decimal(bin_sequences_I3)

    # # 应用 XOR 操作
    dncrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
    dncrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
    dncrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

    I1_prime = reshape_blocks_to_channel(dncrypted_blocks_I1, p)
    I2_prime = reshape_blocks_to_channel(dncrypted_blocks_I2, p)
    I3_prime = reshape_blocks_to_channel(dncrypted_blocks_I3, p)

    decrypted_img = cv2.merge((I1_prime, I2_prime, I3_prime))

    return (True, decrypted_img)


def generate(num: int):
    """
    生成混沌序列    
    """
    env = gym.make('lorenz_transient-v0')
    model = PPO.load(MODEL_DIR, env, verbose=1)
    # model = PPO.load('experiments/exp_lorenz/lorenz_targeting_810k', env, verbose=1)
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

    num += 1500
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


def generate_seed():
    """
    将混沌序列转换为 np.array
    """
    # 严重错误，这个值根本不是切片值
    # num = int((M_image * N_image) / (p * p)) 
    # 正确逻辑：使用向上取整ceil
    # num = ceil(M/p) * ceil(N/p)
    num = ((M_image + p - 1) // p) * ((N_image + p - 1) // p)
    list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, list_x8 = generate(num)

    x1 = np.array(list_x1)
    x2 = np.array(list_x2)
    x3 = np.array(list_x3)
    x4 = np.array(list_x4)
    x5 = np.array(list_x5)
    x6 = np.array(list_x6)
    x7 = np.array(list_x7)
    x8 = np.array(list_x8)

    master_sequence = (x1, x2, x3, x4)
    slave_sequence = (x5, x6, x7, x8)

    return master_sequence, slave_sequence


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


def encrypt_and_decrypt_once(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None, password: str = None):
    if plain_path is None or cipher_path is None or decrypted_path is None:
        raise ValueError("File paths cannot be None.")
    
    master_sequence, slave_sequence = generate_seed()
        
    plain_img = cv2.imread(plain_path)
    if plain_img is None:
        raise FileNotFoundError(f"Unable to load image at {plain_path}")

    # 加密
    cipher_img, decry_key = encrypt(master_sequence, plain_img, password)
    save_image(cipher_img, cipher_path)

    # 解密
    is_success, decry_img = decrypt(slave_sequence, cipher_img, decry_key, password)
    print("Decryption attempt completed.")
    if not is_success:
        print("Error: Decryption failed due to incorrect password.")
        return False
    save_image(decry_img, decrypted_path)

    # 验证解密结果
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


def encrypt_and_decrypt(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None, password: str = None):
    cnt = 0
    while True:
        cnt += 1
        print("********** Round: {} **********".format(cnt))
        flag = encrypt_and_decrypt_once(plain_path, cipher_path, decrypted_path, password=password)
        if flag:
            break
        elif cnt >= 10:
            raise Exception("Decryption failed after 10 attempts.")
            

def process_images_in_folder(source_dir, cipher_dir, decrypted_dir):
    """
    遍历 source_dir 下的所有 diff 图片，进行加密和解密，
    并将结果分别保存到 cipher_dir 和 decrypted_dir。
    如果 decrypted_dir 中已存在同名文件，则跳过（增量更新）。
    """

    global M_image, N_image

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
        # 3. 过滤文件，只处理 tiff (忽略大小写)
        if file_name.lower().endswith('.tiff'):
            
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


            # 【修改点】预读图片并更新全局变量
            # 在进入 encrypt_and_decrypt 之前，强制改变全局尺寸
            temp_img = cv2.imread(plain_path)
            if temp_img is None:
                print(f"[ERROR] Could not read {file_name}, skipping.")
                continue
            
            # 直接修改全局变量，这会影响到后续所有函数的执行（generate_seed, encrypt 等）
            M_image = temp_img.shape[0]  # Height
            N_image = temp_img.shape[1]  # Width
            print(f"[INFO] Global size set to: {M_image}x{N_image} for {file_name}")
            
            
            # 5. 开始处理新图片
            print(f"[ACTION] Processing new image: {file_name}")
            try:
                # 调用你写好的带有重试机制的函数
                encrypt_and_decrypt(plain_path, cipher_path, decrypted_path, password=MY_PASSWORD)
                count_processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {file_name}. Reason: {str(e)}")
                traceback.print_exc()
    
    print("-" * 40)
    print(f"Batch task finished. New Processed: {count_processed}, Skipped: {count_skipped}.")
    pass


if __name__ == "__main__":
    # 定义文件夹路径
    plain_folder = PLAIN_DIR
    cipher_folder = CIPHER_DIR
    decrypted_folder = DECRYPTED_DIR

    # 确保源文件夹存在，否则无法处理
    if os.path.exists(plain_folder):
        try:
            process_images_in_folder(plain_folder, cipher_folder, decrypted_folder)
        except Exception as e:
            print("An error occurred during batch processing:", str(e))
    else:
        print(f"Error: Source directory '{plain_folder}' not found.")

