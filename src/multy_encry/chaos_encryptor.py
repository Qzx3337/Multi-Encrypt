import os
import traceback
import gym
import gym_lorenz
from stable_baselines3 import PPO
import numpy as np
import cv2
import hashlib

# KalmanFilter1D 是一个独立的工具类，可以保留在类外，或者作为静态辅助类
class KalmanFilter1D:
    def __init__(self, initial_estimate, measurement_uncertainty, process_variance):
        self.estimate = initial_estimate
        self.estimate_error = measurement_uncertainty
        self.measurement_uncertainty = measurement_uncertainty
        self.process_variance = process_variance

    def update(self, measurement):
        # ... (保持原代码逻辑不变) ...
        prediction = self.estimate
        # ...
        return self.estimate

class ChaosEncryptor:
    def __init__(self, model_path, password="password987", block_size=8, enable_sync_check=True):
        """
        初始化加密器，加载模型和配置
        """
        self.model_path = model_path
        self.password = password
        self.p = block_size  # 原代码中的 p
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

        # 预加载 RL 模型，避免每次 generate 都读取文件
        print(f"Loading PPO model from {self.model_path}...")
        self.env = gym.make('lorenz_transient-v0')
        self.model = PPO.load(self.model_path, self.env)
        print("Model loaded successfully.")

    # --- 辅助工具方法 (建议转为静态方法或私有方法) ---

    @staticmethod
    def _string_to_initial_value(password: str) -> float:
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
    def _split_channels(image: np.ndarray):
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

    def _logistic_map(self, theta: float, initial_value: float, num_iterations: int) -> list:
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

    def _reshape_sequence_to_Q(self, logistic_sequence: list, height: int, width: int) -> np.ndarray:
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

    # --- 矩阵分块与 DNA 转换工具 ---
    
    def _split_into_blocks(self, matrix: np.ndarray) -> list:
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

    def _reshape_blocks_to_channel(self, blocks: list, height: int, width: int) -> np.ndarray:
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

    def _binary_to_dna(self, binary_blocks, bd):
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
    
    def _dna_to_binary(self, dna_blocks, db):
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

    def _generate_raw_sequence(self, num: int):
        """
        原 generate 函数。
        改动：直接使用 self.model 和 self.env，不需要每次重新 load。

        生成混沌序列    
        """
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
            action, _states = self.model.predict(obs)
            obs, reward, dones, info = self.env.step(action)
            x1 = self.env.get_current()
            x2 = self.env.get_current1()
            x3 = self.env.get_current2()
            x4 = self.env.get_current3()
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


    def _generate_seed(self, height: int, width: int):
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
        list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, list_x8 = self._generate_raw_sequence(num)

        x1 = np.array(list_x1)
        x2 = np.array(list_x2)
        x3 = np.array(list_x3)
        x4 = np.array(list_x4)
        x5 = np.array(list_x5)
        x6 = np.array(list_x6)
        x7 = np.array(list_x7)
        x8 = np.array(list_x8)
        
        # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则 =====

        x1 = (np.mod(np.round(x1), 8) + 1).astype(np.uint8)
        x2 = (np.mod(np.round(x2), 8) + 1).astype(np.uint8)
        x3 = (np.mod(np.round(x3), 8) + 1).astype(np.uint8)
        x4 = (np.mod(np.round(x4), 8) + 1).astype(np.uint8)
            
        x5 = (np.mod(np.round(x5), 8) + 1).astype(np.uint8)
        x6 = (np.mod(np.round(x6), 8) + 1).astype(np.uint8)
        x7 = (np.mod(np.round(x7), 8) + 1).astype(np.uint8)
        x8 = (np.mod(np.round(x8), 8) + 1).astype(np.uint8)

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

            x5 = process_array_with_kalman(x5)
            x6 = process_array_with_kalman(x6)
            x7 = process_array_with_kalman(x7)
            x8 = process_array_with_kalman(x8)

            x5 = process_array(x5)
            x6 = process_array(x6)
            x7 = process_array(x7)
            x8 = process_array(x8)
        

        master_sequence = (x1, x2, x3, x4)
        slave_sequence = (x5, x6, x7, x8)
        # print(f"ytpe of x1: {type(x1)}, shape: {x1.shape}")
        return master_sequence, slave_sequence


    # --- 加密与解密主逻辑 ---

    def encrypt(self, plain_img: np.ndarray):
        """
        输入原图，返回 (密图, 解密Key)
        """
        # 获取动态尺寸
        height, width = plain_img.shape[:2]

        # 1. 生成混沌序列 (支持重试机制)
        master_sequence, _ = self._generate_seed_with_retry(height, width)

        # 2. 拆分通道
        x1, x2, x3, x4 = master_sequence
        blue, green, red = self._split_channels(plain_img)

        # 3. 生成 Q 矩阵 (Seed 逻辑)
        i1, i2, i3 = np.sum(red), np.sum(green), np.sum(blue)
        img_word = str(int(i1) + int(i2) + int(i3))
        raw_seed_str = (self.password if self.password else "") + img_word
        
        initial_value = self._string_to_initial_value(raw_seed_str)
        theta = 3.9999  # Example parameter value
        num_iterations = height * width  # Number of iterations

        logistic_sequence = logistic_map(theta, initial_value, num_iterations - 1)
        # print(len(logistic_sequence))

        Q = reshape_sequence_to_Q(logistic_sequence, height, width)

        # ===== 阶段1.3：通道分别与Q做异或 =====

        blocks_I1 = split_into_blocks(blue, height, width)
        blocks_I2 = split_into_blocks(green, height, width)
        blocks_I3 = split_into_blocks(red, height, width)

        blocks_Q = split_into_blocks(Q, height, width)

        encrypted_blocks_I1 = [np.bitwise_xor(block_I1, block_Q) for block_I1, block_Q in zip(blocks_I1, blocks_Q)]
        encrypted_blocks_I2 = [np.bitwise_xor(block_I2, block_Q) for block_I2, block_Q in zip(blocks_I2, blocks_Q)]
        encrypted_blocks_I3 = [np.bitwise_xor(block_I3, block_Q) for block_I3, block_Q in zip(blocks_I3, blocks_Q)]

        # ===== 阶段2：DNA加密 =====

        # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则(已经在生成时完成) =====

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

        I1_prime = reshape_blocks_to_channel(dec_sequences_I1, height, width)
        I2_prime = reshape_blocks_to_channel(dec_sequences_I2, height, width)
        I3_prime = reshape_blocks_to_channel(dec_sequences_I3, height, width)

        cipher_img = cv2.merge((I1_prime, I2_prime, I3_prime))
        # 生成解密密钥
        # 对于防攻击情况，则不传出 initial_value
        decry_key = (img_word, initial_value)

        return cipher_img, decry_key

    def decrypt(self, cipher_img: np.ndarray, decry_key: tuple):
        """
        输入密图和key，返回 (是否成功, 解密图)
        """
        img_word = decry_key[0]
        # ... (密码校验逻辑) ...

        height, width = cipher_img.shape[:2]
        
        # 生成混沌序列
        # 注意：这里需要重新生成一遍序列，或者你可以选择在 encrypt 时就把 slave_sequence 传出去
        # 原代码逻辑是在 decrypt 里重新调了一次 generate_seed，这需要保证模型 deterministic
        _, slave_sequence = self._generate_seed_with_retry(height, width)
        
        # ... (复制逆向 DNA 解码、XOR、合并逻辑) ...
        
        return True, decrypted_img

    def _generate_seed_with_retry(self, height, width):
        """
        封装了原代码中的同步性检查循环
        """
        if not self.enable_sync_check:
            return self._generate_seed(height, width)

        attempt_num = 5
        for i in range(attempt_num):
            m_seq, s_seq = self._generate_seed(height, width)
            # 简单的同步性检查
            if all(np.array_equal(m, s) for m, s in zip(m_seq, s_seq)):
                return m_seq, s_seq
            print(f"Sync check failed, retrying ({i+1}/{attempt_num})...")
        
        raise RuntimeError("Failed to generate synchronized chaotic sequences.")

    # --- 批处理入口 ---

    def process_folder(self, source_dir, cipher_dir, decrypted_dir):
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
                if plain_img is None: continue

                # 加密
                cipher_img, key = self.encrypt(plain_img)
                cv2.imwrite(cipher_path, cipher_img)

                # 解密 (验证用)
                success, dec_img = self.decrypt(cipher_img, key)
                cv2.imwrite(decrypted_path, dec_img)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                traceback.print_exc()

# --- 使用示例 ---

if __name__ == "__main__":
    # 配置路径
    BASE_DIR = "experiments/w502"
    MODEL_PATH = "experiments/exp_lorenz/lorenz_f2_lr5en5_s1m.zip"
    
    # 1. 实例化类 (模型只加载一次)
    encryptor = ChaosEncryptor(
        model_path=MODEL_PATH, 
        password="my_secure_password",
        block_size=8
    )

    # 2. 调用批处理
    encryptor.process_folder(
        source_dir=os.path.join(BASE_DIR, "hyper_kvasir/plain_img"),
        cipher_dir=os.path.join(BASE_DIR, "hyper_kvasir/cipher_img"),
        decrypted_dir=os.path.join(BASE_DIR, "hyper_kvasir/decrypted_img")
    )
