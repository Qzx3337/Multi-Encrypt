import os
import traceback
import gym
import gym_lorenz
from stable_baselines3 import PPO
import numpy as np
import cv2
import hashlib


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

def process_array_with_kalman(arr, measurement_uncertainty=1e-2, process_variance=1e-5):
    # 使用数组的第一个值作为初始估计值
    initial_estimate = arr[0]
    kf = KalmanFilter1D(initial_estimate, measurement_uncertainty, process_variance)

    # 对数组中的每个元素进行卡尔曼滤波
    filtered_arr = np.array([kf.update(num) for num in arr])

    return filtered_arr


class MultiEncryptor:
    def __init__(self, model_path, password="password987", block_size=8, enable_sync_check=True):
        """
        初始化加密器，加载模型和配置
        """
        self.model_path = model_path
        self.password = password
        self.p = block_size  
        self.enable_sync_check = enable_sync_check
        
        # DNA 编码规则 (属性化)
        self.coding_rules = {
            1: {'00': 'A', '11': 'T', '10': 'C', '01': 'G'},
            2: {'00': 'A', '11': 'T', '01': 'C', '10': 'G'},
            3: {'11': 'A', '00': 'T', '10': 'C', '01': 'G'},
            4: {'11': 'A', '00': 'T', '01': 'C', '10': 'G'},
            5: {'01': 'A', '10': 'T', '00': 'C', '11': 'G'},
            6: {'01': 'A', '10': 'T', '11': 'C', '00': 'G'},
            7: {'10': 'A', '01': 'T', '00': 'C', '11': 'G'},
            8: {'10': 'A', '01': 'T', '11': 'C', '00': 'G'}
        }

    # --- 辅助工具方法 ---

    @staticmethod
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


    @staticmethod
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


    def logistic_map(self, theta: float, initial_value: float, num_iterations: int) -> list:
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


    def reshape_sequence_to_Q(self, logistic_sequence: list, height: int, width: int) -> np.ndarray:
        """
        将混沌序列转换为掩码矩阵 Q。
        Args:
            logistic_sequence (list of float): 混沌序列列表，其中的变量为浮点数。
            width (int): 图像的宽度。
            height (int): 图像的高度。
        Returns:
            Q (np.ndarray): 掩码矩阵, 形状为 (width, height).
        """
        # 将列表转换为 NumPy 数组
        K1_array = np.array(logistic_sequence)

        # 量化：(0, 1)浮点数转换为 0-255 范围内的整数
        K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)

        # 一维向量重塑为 (width, height) 形状的矩阵 Q
        Q = K1_prime.reshape(height, width)
        return Q


    def check_img_pixel(self, plain_img: np.ndarray, decrypted_path: np.ndarray) -> bool:
        """
        检查解密图像与明文图像的像素差异，确保加密有效性。
        
        Args:
            plain_img (np.ndarray): 明文图像矩阵.
            decrypted_path (np.ndarray): 解密结果图像矩阵.
        Returns:
            is_different (bool): 如果两张图像在像素上完全一致 True，否则返回 False.
        """
        if plain_img.shape != decrypted_path.shape:
            print("Error: Image shapes do not match for pixel check.")
            return False
        
        difference = np.abs(plain_img.astype(np.int16) - decrypted_path.astype(np.int16))
        num_different_pixels = np.count_nonzero(difference)
        
        if num_different_pixels == 0:
            print("Pixel difference check passed: Pixel values match exactly.")
            return True
        else:
            print(f"Pixel difference check failed: {num_different_pixels} pixels differ.")
            return False

    # --- 矩阵分块与 DNA 转换工具 ---
    
    def split_into_blocks(self, matrix: np.ndarray) -> list:
        """
        将二维矩阵拆分为一系列 p x p 的小块。
        
        Args:
            matrix (np.ndarray): 图像的某个通道(W x H)
        Returns:
            blocks (list of np.ndarray): 包含所有 p x p 小块的列表.
        """
        blocks = []
        height, width = matrix.shape
        for i in range(0, height, self.p):  # 确保不超出边界
            for j in range(0, width, self.p):
                block = matrix[i:i + self.p, j:j + self.p]
                blocks.append(block)
        return blocks


    def reshape_blocks_to_channel(self, blocks: list, height: int, width: int) -> np.ndarray:
        """
        将一系列 p x p 的小块重新组合为一个二维矩阵。
        Args:
            blocks (list of np.ndarray): 包含所有 p x p 小块的列表.
            width (int): 图像的宽度。
            height (int): 图像的高度。
        Returns:
            reshaped_matrix (np.ndarray): 重新组合后的二维矩阵，形状为 (width, height).
        """
        reshaped_matrix = np.zeros((height, width), dtype=np.uint8)
        index = 0
        for i in range(0, height, self.p):
            for j in range(0, width, self.p):
                reshaped_matrix[i:i + self.p, j:j + self.p] = blocks[index]
                index += 1
        return reshaped_matrix
    
    
    def convert_to_8bit_binary(self, matrix_block_all: list) -> list:
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


    def convert_binary_to_decimal(self, matrix_block_all):
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


    def binary_to_dna(self, binary_blocks, bd):
        """
        根据给定的二进制块、x1值和编码规则，将二进制字符串转换为DNA序列，并保持三维列表结构。

        Args:
            binary_blocks (list of list of list of str): 三维列表，每个元素是一个二进制字符串块。
            x1_bd (list of int): 每个块对应的x1值列表。

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
            rule = self.coding_rules.get(bd[i])
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


    def dna_to_binary(self, dna_blocks, db):
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
            rule = self.coding_rules.get(db[i])
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

    # --- 核心混沌序列生成逻辑 ---

    def generate_raw_sequence(self, num: int):
        """
        原 generate 函数。
        改动：直接使用 self.model 和 self.env，不需要每次重新 load。

        生成混沌序列    
        """
        env = gym.make('lorenz_transient-v0')
        model = PPO.load(self.model_path, env, verbose=1)
        obs = env.reset()

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
        list_obss = (list_obs1, list_obs2, list_obs3, list_obs4, list_obs5, list_obs6, list_obs7, list_obs8)
        return list_obss


    def sequence_quantization(self, height: int, width: int):
        """
        原 generate_seed 函数。
        将混沌序列转换为 np.array .
        并且直接输出量化后的序列

        Args:
            height (int): 图像的高度.
            width (int): 图像的宽度.
            p (int): 切片的大小.
        Returns:
            master_sequence (tuple of np.ndarray): 主序列，包含四个量化后的混沌序列.
            slave_sequence (tuple of np.ndarray): 从序列，包含四个量化后的混沌序列.
        """

        # 使用向上取整ceil
        # num = ceil(M/p) * ceil(N/p)
        num = ((height + self.p - 1) // self.p) * ((width + self.p - 1) // self.p)
        list_xs = self.generate_raw_sequence(num)

        xs = [np.array(list_x) for list_x in list_xs]
        
        # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则 =====

        xs = [(np.mod(np.round(xi), 8) + 1).astype(np.uint8) for xi in xs]

        def process_array(arr):
            processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
            return processed_arr

        # 滤波与量化掺杂进行
        for i in range(15):
            xs = [process_array_with_kalman(xi) for xi in xs]
            xs = [process_array(xi) for xi in xs]

        master_sequence = (xs[0], xs[1], xs[2], xs[3])
        slave_sequence = (xs[4], xs[5], xs[6], xs[7])
        # print(f"ytpe of x1: {type(x1)}, shape: {x1.shape}")
        return master_sequence, slave_sequence

    # --- 加密与解密主逻辑 ---

    def encrypt(self, master_sequence, plain_img: np.ndarray):
        x1, x2, x3, x4 = master_sequence

        # ===== 阶段1：Q置乱 =====
        # ===== 阶段1.1：明文的通道拆分=====

        channels = cv2.split(plain_img)
        height, width = plain_img.shape[:2]

        # ===== 阶段1.2：生成掩码矩阵Q =====

        # 原本的初始值生成方式：基于明文信息提取
        i1 = np.sum(channels[0])  # blue
        i2 = np.sum(channels[1])  # green
        i3 = np.sum(channels[2])  # red
        # initial_value = (i1 + i2) / (255 * height * width * 2) 

        # 带密码的初始值生成方式：图像信息与用户密码混合
        img_word = str(int(i1) + int(i2) + int(i3))

        # 检查password必须是字符串
        if isinstance(self.password, str):
            raw_seed_str = self.password + img_word
        elif self.password is None:
            raw_seed_str = img_word
        else:
            raise ValueError("Password must be a string or None.")
        
        initial_value = self.string_to_initial_value(raw_seed_str)
        theta = 3.9999  # Example parameter value
        num_iterations = height * width  # Number of iterations

        logistic_sequence = self.logistic_map(theta, initial_value, num_iterations - 1)
        # print(len(logistic_sequence))

        Q = self.reshape_sequence_to_Q(logistic_sequence, height, width)

        # ===== 阶段1.3：通道分别与Q做异或 =====

        channels_blocks = [self.split_into_blocks(ch) for ch in channels]
        blocks_Q = self.split_into_blocks(Q)

        # diffused_channels = []
        # for ch_blocks in channels_blocks:
        #     diffused_ch = [np.bitwise_xor(block_I, block_Q) for block_I, block_Q in zip(ch_blocks, blocks_Q)]
        #     diffused_channels.append(diffused_ch)

        diffused_channels = []
        for blocks in channels_blocks:
            diffused_blocks = []
            for block_I, block_Q in zip(blocks, blocks_Q):
                diffused_block = np.bitwise_xor(block_I, block_Q)
                diffused_blocks.append(diffused_block)
            diffused_channels.append(diffused_blocks)

        # ===== 阶段2：DNA加密 =====

        # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则(已经在生成时完成) =====

        # ===== 阶段2.2：各通道转换为8位二进制字符串 =====
        bin_channels = [self.convert_to_8bit_binary(ch) for ch in diffused_channels]

        # ===== 阶段2.3：基于DNA编解码的加密 =====

        dna_channels = [self.binary_to_dna(bin_ch, x1) for bin_ch in bin_channels]
        bin_channels = [self.dna_to_binary(dna_ch, x2) for dna_ch in dna_channels]
        dna_channels = [self.binary_to_dna(bin_ch, x3) for bin_ch in bin_channels]
        bin_channels = [self.dna_to_binary(dna_ch, x4) for dna_ch in dna_channels]

        # ===== 阶段3：保存图像与生成密钥 =====

        channels_blocks = [self.convert_binary_to_decimal(bin_ch) for bin_ch in bin_channels]

        channels = [self.reshape_blocks_to_channel(ch, height, width) for ch in channels_blocks]

        cipher_img = cv2.merge(channels)
        # 生成解密密钥
        # 对于防攻击情况，则不传出 initial_value
        decry_key = (img_word, initial_value)

        return cipher_img, decry_key


    def decrypt(self, slave_sequence: tuple, cipher_img: np.ndarray, decry_key: tuple):
        """
        解密算法，相当于加密的逆过程。

        Args:
            slave_sequence (tuple of np.ndarray): 从序列，包含四个量化后的混沌序列.
            cipher_img (np.ndarray): 密文图像矩阵，形状为 (M, N, C)。
            decry_key (tuple): 解密密钥，包含 img_word (str) 和 initial_value_of_Q (float)(可选)。
            password (str, optional): 用户输入的密码字符串。如果没有密码，则为 None。

        Returns:
            is_success (bool): 解密是否成功的标志。
            decrypted_img (np.ndarray or None): 解密后的图像矩阵，如果解密失败则为 None。
        """

        # 对密钥解包
        # img_word, initial_value_of_Q = decry_key
        # 为防止攻击则不传入 initial_value_of_Q
        img_word = decry_key[0]

        # 验证用户是否输入了正确的密码
        if isinstance(self.password, str):
            raw_seed_str = self.password + img_word
        elif self.password is None:
            raw_seed_str = img_word
        else:
            raise ValueError("Password must be a string or None.")
        calculated_initial_value = self.string_to_initial_value(raw_seed_str)
        # 对于防攻击情况
        initial_value_of_Q = calculated_initial_value
        # 对于测试中间件正确性的情况
        # if not np.isclose(expected_initial_value, initial_value_of_Q):
        #     print("Error: Incorrect password provided for decryption.")
        #     return (False, None)
        
        # 密文通道拆分
        # blue_c, green_c, red_c = split_channels(cipher_img)
        # height, width = blue_c.shape
        cipher_channels = cv2.split(cipher_img)
        height, width = cipher_img.shape[:2]
        
        # 重新生成 Q 矩阵
        theta = 3.9999
        num_iterations = height * width  # Number of iterations
        logistic_sequence = self.logistic_map(theta, initial_value_of_Q, num_iterations - 1)
        Q = self.reshape_sequence_to_Q(logistic_sequence, height, width)
        blocks_Q = self.split_into_blocks(Q)

        # 量化混沌序列
        x5, x6, x7, x8 = slave_sequence
        
        # 分块
        cipher_channel_blocks = [self.split_into_blocks(ch) for ch in cipher_channels]

        # DNA解密

        # 转换为8位二进制字符串
        bin_channel_blocks = [self.convert_to_8bit_binary(ch) for ch in cipher_channel_blocks]

        # 基于DNA编解码的解密
        dna_channel_blocks = [self.binary_to_dna(bin_ch, x8 ) for bin_ch in bin_channel_blocks]
        bin_channel_blocks = [self.dna_to_binary(dna_ch, x7 ) for dna_ch in dna_channel_blocks]
        dna_channel_blocks = [self.binary_to_dna(bin_ch, x6 ) for bin_ch in bin_channel_blocks]
        bin_channel_blocks = [self.dna_to_binary(dna_ch, x5 ) for dna_ch in dna_channel_blocks]

        # 转换回十进制整数
        channels_blocks = [self.convert_binary_to_decimal(bin_ch) for bin_ch in bin_channel_blocks]

        # 解密Q置乱
        dncrypted_channels_blocks = []
        for channel_blocks in channels_blocks:
            dncrypted_channel = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(channel_blocks, blocks_Q)]
            dncrypted_channels_blocks.append(dncrypted_channel)

        decrypted_channels = [self.reshape_blocks_to_channel(ch, height, width) for ch in dncrypted_channels_blocks]

        decrypted_img = cv2.merge(decrypted_channels)

        return (True, decrypted_img)


    def generate_with_retry(self, height, width):
        """
        封装了原代码中的同步性检查循环
        """
        if not self.enable_sync_check:
            return self.sequence_quantization(height, width)

        attempt_num = 5
        for i in range(attempt_num):
            m_seq, s_seq = self.sequence_quantization(height, width)
            # 简单的同步性检查
            if all(np.array_equal(m, s) for m, s in zip(m_seq, s_seq)):
                return m_seq, s_seq
            print(f"Sync check failed, retrying ({i+1}/{attempt_num})...")
        
        raise RuntimeError("Failed to generate synchronized chaotic sequences.")

    # --- 处理流程入口 ---
    def process_single_matrix(self, plain_img: np.ndarray)-> tuple:
        """
        Args:
            plain_img (np.ndarray): 明文图像矩阵，形状为 (M, N, C)。
        Returns:
            cipher_img (np.ndarray): 密文图像矩阵，形状为 (M, N, C)。
            dec_img (np.ndarray): 解密后的图像矩阵，形状为 (M, N, C)。
        说明：该函数仅用于测试中间件的正确性，不建议在批处理时使用。
        """
        height, width = plain_img.shape[:2]
        master_seq, slave_seq = self.generate_with_retry(height, width)

        # 加密
        cipher_img, decry_key = self.encrypt(master_seq, plain_img)

        # 解密 (验证用)
        success, dec_img = self.decrypt(slave_seq, cipher_img, decry_key)

        return cipher_img, dec_img


    def process_multimodal_folder(self):
        pass

    def process_unimodal_folder(self, source_dir, cipher_dir, decrypted_dir):
        """
        批处理文件夹
        """
        os.makedirs(cipher_dir, exist_ok=True)
        os.makedirs(decrypted_dir, exist_ok=True)
        
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.tiff', '.tif'))]
        
        for file_name in files:
            plain_path = os.path.join(source_dir, file_name)
            cipher_path = os.path.join(cipher_dir, file_name)
            decrypted_path = os.path.join(decrypted_dir, file_name)

            if os.path.exists(decrypted_path):
                print(f"[SKIP] {file_name}")
                continue

            print(f"Processing {file_name}...")
            
            try:
                # 读取
                plain_img = cv2.imread(plain_path)
                if plain_img is None: raise ValueError("Failed to read image.")
                
                # 生成混沌序列
                height, width = plain_img.shape[:2]
                master_seq, slave_seq = self.generate_with_retry(height, width)

                # 加密
                cipher_img, key = self.encrypt(master_seq, plain_img)
                cv2.imwrite(cipher_path, cipher_img)
                print(f"成功保存 {cipher_img.shape[2]} 通道图像至: {cipher_path}")

                # 解密
                incorrect_key, dec_img = self.decrypt(slave_seq, cipher_img, key)
                cv2.imwrite(decrypted_path, dec_img)
                print(f"成功保存 {dec_img.shape[2]} 通道图像至: {decrypted_path}")

                # 验证
                if not self.check_img_pixel(plain_img, dec_img):
                    print(f"Warning: Decrypted image does not match the original for {file_name}.")
                print()


            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                traceback.print_exc()

# --- 使用示例 ---

if __name__ == "__main__":
    
    # 配置路径
    EXPERMENT_DIR = "experiments/w502"
    DATABASE_DIR = os.path.join(EXPERMENT_DIR, "hyper_kvasir")
    MODEL_PATH = "experiments/exp_lorenz/lorenz_f2_lr5en5_s1m.zip"
    
    # # 1. 实例化类
    # encryptor = ChaosEncryptor(
    #     model_path=MODEL_PATH, 
    #     password="my_secure_password",
    #     block_size=8,
    #     enable_sync_check=True
    # )

    # # 2. 调用批处理
    # encryptor.process_unimodal_folder(
    #     source_dir=os.path.join(DATABASE_DIR, "plain_img"),
    #     cipher_dir=os.path.join(DATABASE_DIR, "cipher_img"),
    #     decrypted_dir=os.path.join(DATABASE_DIR, "decrypted_img")
    # )
