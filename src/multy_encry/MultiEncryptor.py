import os
import traceback
import gym
import gym_lorenz
from stable_baselines3 import PPO
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import hashlib


# 配置路径
# data\zky\images
DATABASE_NAME = "zky" # 数据集变动改这里
EXPERMENT_DIR = os.path.join("experiments", "w503") # 做新实验改这里  
PLAIN_IMGS_DIR = os.path.join("data", DATABASE_NAME, "images")
CIPHER_IMGS_DIR = os.path.join(EXPERMENT_DIR, DATABASE_NAME, "cipher")
DECRYPTED_IMGS_DIR = os.path.join(EXPERMENT_DIR, DATABASE_NAME, "decry", "images")
DRAW_DIR = os.path.join(EXPERMENT_DIR, "draw_disp")

MODEL_PATH = os.path.join("experiments", "exp_lorenz", "lorenz_f2_lr5en5_s1m.zip")

ENABLE_SYNC_CHECK = True
KALMAN_FLITER_TIMES = 1
CLIP_STEP = 3000


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
        
        if not os.path.exists(model_path):
            raise RuntimeError("[error] 模型文件不存在")
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


    @staticmethod
    def visualize_chaos_seq(chaos_seq, save_path, max_display_len=200):
        """
        可视化四维混沌系统的离散整数轨迹序列。
        
        参数:
            chaos_seq (tuple): 包含四个 np.ndarray 的元组 (x1, x2, x3, x4)。
                            每个数组包含区间 [1, 8] 内的整数。
            save_path (str): 图片保存的完整路径（例如 'chaos_plot.png'）。
            max_display_len (int): 限制绘图显示的最大长度。
                                如果序列超过此长度，只画前 max_display_len 个点，
                                以便清晰观察细节。设为 None 则画全部。
        """
        
        # 1. 解包数据
        x1, x2, x3, x4 = chaos_seq
        total_len = len(x1)
        
        # 2. 确定实际绘制的长度
        if max_display_len is None or max_display_len > total_len:
            plot_len = total_len
            display_x = x1
        else:
            plot_len = max_display_len
            # 切片，只取前 plot_len 个数据
            x1 = x1[:plot_len]
            x2 = x2[:plot_len]
            x3 = x3[:plot_len]
            x4 = x4[:plot_len]
        
        # 生成时间轴 (x轴)
        t = np.arange(plot_len)
        
        # 3. 创建画布：4行1列
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # 定义颜色和标签，方便区分
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 蓝、橙、绿、红
        labels = ['$x_1$', '$x_2$', '$x_3$', '$x_4$']
        data_list = [x1, x2, x3, x4]
        
        # 4. 循环绘制每个维度
        for i, ax in enumerate(axes):
            # 使用 step 绘图，where='post' 表示阶梯在点之后跳变，适合离散状态
            ax.step(t, data_list[i], where='post', color=colors[i], linewidth=1.5, label=labels[i])
            
            # 散点图辅助：在每个点的位置画一个小点，方便看清具体位置（可选）
            ax.scatter(t, data_list[i], color=colors[i], s=10, alpha=0.6)
            
            # 设置 Y 轴范围和刻度（因为只有1-8）
            ax.set_ylim(0.5, 8.5)
            ax.set_yticks(range(1, 9))  # 强制显示 1 到 8 的刻度
            ax.set_ylabel(f'State ({labels[i]})', fontsize=12)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right')

        # 5. 设置整体标签
        axes[-1].set_xlabel('Time Step (t)', fontsize=12)
        plt.suptitle(f'4D Chaos Sequence Visualization (First {plot_len} steps)', fontsize=16)
        
        # 调整布局防止重叠
        plt.tight_layout()
        
        # 6. 保存到磁盘
        try:
            # 确保目录存在
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" 混沌序列量化结果图已成功保存至: {save_path}")
        except Exception as e:
            print(f"❌ 混沌序列量化结果图保存失败: {e}")
        finally:
            plt.close() # 关闭画布释放内存

    @staticmethod
    def visualize_chaos_continuous(chaos_seq, save_path, max_display_len=1000):
        """
        可视化四维混沌系统的连续浮点数轨迹序列。
        
        参数:
            chaos_seq (tuple): 包含四个 np.ndarray 的元组 (x1, x2, x3, x4)。
                            每个数组包含浮点数 (float)。
            save_path (str): 图片保存的完整路径（例如 'chaos_float.png'）。
            max_display_len (int): 限制绘图显示的最大长度。
                                对于连续波形，建议显示长度稍长一点（如1000）以观察趋势。
                                设为 None 则画全部。
        """
        
        # 1. 解包数据
        x1, x2, x3, x4 = chaos_seq
        total_len = len(x1)
        
        # 2. 确定实际绘制的长度
        # 如果数据太长，截取前一部分，否则线条会挤在一起看不清震荡细节
        if max_display_len is None or max_display_len > total_len:
            plot_len = total_len
            display_x1, display_x2, display_x3, display_x4 = x1, x2, x3, x4
        else:
            plot_len = max_display_len
            display_x1 = x1[:plot_len]
            display_x2 = x2[:plot_len]
            display_x3 = x3[:plot_len]
            display_x4 = x4[:plot_len]
        
        data_list = [display_x1, display_x2, display_x3, display_x4]
        t = np.arange(plot_len)
        
        # 3. 创建画布：4行1列
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # 定义颜色和标签
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 蓝、橙、绿、红
        labels = ['$x_1(t)$', '$x_2(t)$', '$x_3(t)$', '$x_4(t)$']
        
        # 4. 循环绘制每个维度
        for i, ax in enumerate(axes):
            # --- 核心修改：使用 plot 绘制连续曲线 ---
            # linewidth 设置为 1.2，既能看清细节又不会太粗
            ax.plot(t, data_list[i], color=colors[i], linewidth=1.2, label=labels[i])
            
            # --- 核心修改：Y轴自适应 ---
            # 不再强制设置 set_yticks，让 matplotlib 根据浮点数范围自动适配
            # 仅保留网格线以便观察数值大小
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            ax.set_ylabel(f'Value', fontsize=10)
            ax.legend(loc='upper right', frameon=True)
            
            # 可选：显示当前维度的最大最小值，方便分析范围
            d_min, d_max = np.min(data_list[i]), np.max(data_list[i])
            # 在左上角标注数值范围
            ax.text(0.01, 0.9, f'Range: [{d_min:.2f}, {d_max:.2f}]', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 5. 设置整体标签
        axes[-1].set_xlabel('Time Step (t)', fontsize=12)
        plt.suptitle(f'4D Continuous Chaos Trajectory (First {plot_len} steps)', fontsize=16)
        
        # 调整布局
        plt.tight_layout()
        
        # 6. 保存到磁盘
        try:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 连续轨迹图已成功保存至: {save_path}")
        except Exception as e:
            print(f"❌ 保存图片失败: {e}")
        finally:
            plt.close()
            

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


    # def reshape_sequence_to_Q(self, logistic_sequence: list, height: int, width: int) -> np.ndarray:
    #     """
    #     将混沌序列转换为掩码矩阵 Q。
    #     Args:
    #         logistic_sequence (list of float): 混沌序列列表，其中的变量为浮点数。
    #         width (int): 图像的宽度。
    #         height (int): 图像的高度。
    #     Returns:
    #         Q (np.ndarray): 掩码矩阵, 形状为 (width, height).
    #     """
    #     # 将列表转换为 NumPy 数组
    #     K1_array = np.array(logistic_sequence)

    #     # 量化：(0, 1)浮点数转换为 0-255 范围内的整数
    #     K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)

    #     # 一维向量重塑为 (width, height) 形状的矩阵 Q
    #     Q = K1_prime.reshape(height, width)
    #     return Q
    
    def reshape_sequence_to_Q(self, logistic_sequence: list, height: int, width: int, block_size: int = 8) -> np.ndarray:
        """
        将混沌序列按照 8x8 块进行交织填充，以降低空间相关性。
        填充逻辑：先填充所有块的(0,0)，再填充所有块的(0,1)...以此类推。
        """
        # 1. 基础转换与量化 (保持原逻辑)
        K1_array = np.array(logistic_sequence)
        K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)

        # 2. 计算 Pad 后的尺寸
        # 为了方便矩阵运算，我们将图像宽高扩展为 8 的倍数
        h_blocks = math.ceil(height / block_size)
        w_blocks = math.ceil(width / block_size)
        
        h_pad = h_blocks * block_size
        w_pad = w_blocks * block_size
        
        total_pixels_needed = h_pad * w_pad
        num_blocks = h_blocks * w_blocks
        pixels_per_block = block_size * block_size

        # 3. 序列长度检查与截取
        # 如果序列不够长，需要循环补全 (防止越界)
        if len(K1_prime) < total_pixels_needed:
            repeats = math.ceil(total_pixels_needed / len(K1_prime))
            K1_prime = np.tile(K1_prime, repeats)
            print("logistic_sequence 不够长")
        
        # 截取刚好填满 Pad 后图像所需的长度
        data_flat = K1_prime[:total_pixels_needed]

        # 4. 核心变换逻辑 (使用 NumPy 维度操作代替循环)
        
        # 第一步：Reshape 成 (块内像素总数, 块的总数)
        # 形状: (64, num_blocks)
        # 这样 data[0] 就包含了所有块的第1个像素，data[1] 包含所有块的第2个像素...
        # 符合你要求的 "序列第一批值分别赋给每个小块的左上角"
        step1 = data_flat.reshape(pixels_per_block, num_blocks)

        # 第二步：展开维度
        # 形状: (8, 8, h_blocks, w_blocks) 
        # 维度含义: (块内行 u, 块内列 v, 块的行索引 br, 块的列索引 bc)
        step2 = step1.reshape(block_size, block_size, h_blocks, w_blocks)

        # 第三步：维度置换 (Transpose)
        # 我们需要的最终图像顺序是: (块的行索引, 块内行, 块的列索引, 块内列)
        # 对应的轴变换为: (2, 0, 3, 1)
        step3 = step2.transpose(2, 0, 3, 1)

        # 第四步：合并维度还原为 2D 图像
        # 形状: (h_pad, w_pad)
        Q_padded = step3.reshape(h_pad, w_pad)

        # 5. 裁剪回原始尺寸 (去除 Padding)
        Q = Q_padded[:height, :width]

        return Q

    @staticmethod
    def reshape_sequence_to_global_mask(logistic_seq:list, shape: list)-> np.ndarray:
        seq_arr = np.array(logistic_seq)
        quantlized_seq = np.mod(np.round(seq_arr * 1000), 256).astype(np.uint8)
        Q = quantlized_seq.reshape(shape)
        return Q
    
    @staticmethod
    def channel_diffusion(channels: list, Q: np.ndarray):
        """
        分别对图像的各个通道进行掩码置乱加通道间扩散。
        
        Note:
            I1' = I1 XOR Q XOR I2
            I2' = I2 XOR Q XOR I3
            ...
            In' = In XOR Q XOR I1' (注意这个最后一次的I1'是已经更新过的)

        Args:
            channels (list of np.ndarray): 图像的各个通道组成列表.
            Q (np.ndarray): 掩码矩阵.
        Returns:
            diffused_channels (list of np.ndarray): 置乱和扩散后的图像通道列表.
        """

        n = len(channels)
        if n < 2: return # 至少需要2个通道才能进行通道间扩散

        # 1. 正向遍历处理前 n-1 个通道
        # 公式: New_Ch[i] = Old_Ch[i] ^ Q ^ Old_Ch[i+1]
        # 注意：由于我们还没有处理 i+1，所以 channels[i+1] 此时还是原始值，符合公式
        for i in range(n - 1):
            # channels[i] ^= Q
            channels[i] ^= channels[i+1]

        # 2. 处理最后一个通道 (闭环)
        # 公式: New_Ch[n-1] = Old_Ch[n-1] ^ Q ^ New_Ch[0]
        # 注意：此时 channels[0] 已经在第1步被修改过了，符合公式要求的 "更新过的 I1'"
        # channels[n - 1] ^= Q
        channels[n - 1] ^= channels[0]

        if Q.ndim == 2:
            for i in range(n):
                channels[i] ^= Q
        elif Q.ndim == 3:
            for i in range(n):
                channels[i] ^= Q[:, :, i]
        else:
            print(f"Q.ndim is: {Q.ndim}")
            raise ValueError("不合法的掩码矩阵Q")

        return channels


    @staticmethod
    def channel_inverse_diffusion(channels: list, Q: np.ndarray):
        """
        [逆向] 通道间扩散逆运算
        
        注释：
        解密逻辑必须与加密逻辑顺序完全相反：
        1. 先利用 New_Ch[0] 恢复 Old_Ch[n-1]
        2. 再倒序利用恢复好的 Old_Ch[i+1] 恢复 Old_Ch[i]

        Args:
            channels (list of np.ndarray): 图像的各个通道组成列表.
            Q (np.ndarray): 掩码矩阵.
        Returns:
            diffused_channels (list of np.ndarray): 置乱和扩散后的图像通道列表.

        """
        n = len(channels)
        if n < 2: return

        if Q.ndim == 2:
            for i in range(n):
                channels[i] ^= Q
        elif Q.ndim == 3:
            for i in range(n):
                channels[i] ^= Q[:, :, i]
        else:
            print(f"Q.ndim is: {Q.ndim}")
            raise ValueError("不合法的掩码矩阵Q")

        # 1. 首先恢复最后一个通道
        # 加密时: C[n-1] = C[n-1] ^ Q ^ C[0](new)
        # 解密时: C[n-1] = C[n-1](new) ^ Q ^ C[0](new)
        # 原理: A ^ B = C  =>  A = C ^ B
        # channels[n - 1] ^= Q
        channels[n - 1] ^= channels[0]

        # 2. 倒序恢复前 n-1 个通道 (从 n-2 遍历到 0)
        # 加密时: C[i] = C[i] ^ Q ^ C[i+1](old)
        # 解密时: C[i] = C[i](new) ^ Q ^ C[i+1](old)
        # 注意：因为我们是倒序处理，计算 C[i] 时，C[i+1] 已经被上一步循环恢复成 Old 值了
        for i in range(n - 2, -1, -1):
            # channels[i] ^= Q
            channels[i] ^= channels[i+1]
        return channels


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

        num += CLIP_STEP
        for i in range(num):
            action, _states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            x1 = env.get_current()
            x2 = env.get_current1()
            x3 = env.get_current2()
            x4 = env.get_current3()
            if i > (CLIP_STEP - 1):
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
        list_obss = [list_obs1, list_obs2, list_obs3, list_obs4, list_obs5, list_obs6, list_obs7, list_obs8]
        list_xs = [np.array(list_x) for list_x in list_obss]
        return list_xs


    def deadzone_quantized_sequence(self, target_length: int, epsilon_max: float):
        """
        带死区的自适应长度量化器
        
        Args:
            target_length (int): 最终需要的有效量化数据长度
            epsilon_max (float): 系统的最大同步误差
            
        Returns:
            list of np.ndarray: 包含8个向量的列表，每个向量长度均为 target_length
            [m1, m2, m3, m4, s1, s2, s3, s4]
        """
        
        # 1. 设定物理参数
        # 保护带阈值
        delta = 1.25 * epsilon_max
        # 量化步长 
        # 可根据动态范围更改这个倍数
        Q = 5 * delta
        
        # 2. 初始化结果容器 (8个通道)
        # 使用列表作为动态缓冲区
        quantized_channels = [[] for _ in range(8)]
        
        # 3. 循环生成，直到所有通道都填满 target_length
        # 注意这里采取的方案是不进行坐标对齐的，效率更高
        # 所有量化后的坐标值已经失去空间位置的语义，但是对加密任务无影响
        # 如果对其坐标应该四个序列同时弃取
        while True:
            # 检查当前各通道的最小长度
            current_lens = [len(c) for c in quantized_channels]
            min_len = min(current_lens)
            
            # 终止条件：所有通道都攒够了数据
            if min_len >= target_length:
                break
            
            # 4. 计算需要补货的数量
            # 预估效率：效率 = 2 * delta / Q
            # 考虑到随机性，我们请求 1.2 倍于缺口的量，防止频繁调用小batch
            needed = target_length - min_len
            batch_size = int(needed / (2 * delta / Q) * 1.2) + 10  # +10 是为了防止 needed 很小时 batch 为 0
            
            # 5. 调用你的原始序列生成器
            # raw_data shape: (8, batch_size)
            raw_data = self.generate_raw_sequence(batch_size)
            
            # 6. 对四个维度分别进行“主系统判决”
            for dim in range(4):
                # 获取当前维度的主动系统数据 (Master) 和从动系统数据 (Slave)
                # 假设 raw_data 的前4行是 Master，后4行是 Slave
                m_vec = raw_data[dim]
                s_vec = raw_data[dim + 4]
                
                # --- 核心逻辑：基于 Master 的死区掩码计算 ---
                
                # 计算 Master 相对于量化栅格的偏移
                # 注意：这里保持原始动态范围，不做归一化
                remainder = m_vec % Q
                
                # 生成布尔掩码 (Mask)
                # 有效条件：余数必须在 [delta, Q - delta] 之间
                # 这样保证了 Master 距离上边界和下边界至少都有 delta 的距离
                valid_mask = (remainder >= delta) & (remainder <= (Q - delta))
                
                # --- 关键：同时应用 Mask 到 Master 和 Slave ---
                # 即使 Slave 的数据是好的，只要 Master 丢了，Slave 必须跟着丢
                m_valid = m_vec[valid_mask]
                s_valid = s_vec[valid_mask]
                
                # --- 量化 ---
                # 直接向下取整并转为整数
                q_m = np.floor(m_valid / Q).astype(int)
                q_s = np.floor(s_valid / Q).astype(int)
                
                # 将结果追加到对应的缓冲区
                quantized_channels[dim].extend(q_m)
                quantized_channels[dim + 4].extend(q_s)
        
        # 7. 截断与格式化输出
        # 因为不同维度的随机性，某些维度可能比其他维度跑得快，长于 target_length
        # 我们统一截断到 target_length 并转换为 numpy 数组
        final_output = []
        for ch_data in quantized_channels:
            final_output.append(np.array(ch_data[:target_length], dtype=int))
            
        return final_output

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
        
        # list of np.ndarray
        quantized_seq = self.deadzone_quantized_sequence(num, 0.4)

        quantized_seq = [arr % 8 + 1 for arr in quantized_seq]

        master_sequence = quantized_seq[0:4]
        slave_sequence = quantized_seq[4:8]

        # visualize_quantized_result(np.arange(num), xs[0], quantized_seq[0], 0, os.path.join(DRAW_DIR, 'dim0_cyclic_check.png'))
        # visualize_quantized_result(np.arange(num), xs[1], quantized_seq[1], 1, os.path.join(DRAW_DIR, 'dim1_cyclic_check.png'))
        # visualize_quantized_result(np.arange(num), xs[2], quantized_seq[2], 2, os.path.join(DRAW_DIR, 'dim2_cyclic_check.png'))
        # visualize_quantized_result(np.arange(num), xs[3], quantized_seq[3], 3, os.path.join(DRAW_DIR, 'dim3_cyclic_check.png'))

        
        # # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则 =====
        # # 量化
        # xs = [(np.mod(np.round(xi), 8) + 1).astype(np.uint8) for xi in xs]

        # # 量化函数
        # def quantlize_array(arr):
        #     processed_arr = (np.mod(arr, 8) + 1).astype(np.uint8)
        #     return processed_arr

        # # 滤波与量化掺杂进行
        # # for i in range(15):
        # for i in range(KALMAN_FLITER_TIMES):
        #     xs = [process_array_with_kalman(xi) for xi in xs]
        #     xs = [quantlize_array(xi) for xi in xs]

        # master_sequence = (xs[0], xs[1], xs[2], xs[3])
        # slave_sequence = (xs[4], xs[5], xs[6], xs[7])


        return master_sequence, slave_sequence

    # --- 加密与解密主逻辑 ---

    def encrypt(self, master_sequence, plain_img: np.ndarray):
        """
        加密算法主流程：1. 掩码矩阵Q置乱 2. 通道间扩散(TODO) 3. DNA加密 4. 保存图像与生成密钥
        Args:
            master_sequence (tuple of np.ndarray): 主序列，包含四个量化后的混沌序列.
            plain_img (np.ndarray): 明文图像矩阵，形状为 (M, N, C)。
        Returns:
            cipher_img (np.ndarray): 密文图像矩阵，形状为 (M, N, C)。
            decry_key (tuple): 解密密钥，包含 img_word (str) 。
        """
        x1, x2, x3, x4 = master_sequence

        # ===== 阶段1：Q置乱 =====
        # ===== 阶段1.1：明文的通道拆分=====

        channels = list(cv2.split(plain_img))

        height, width, depth = plain_img.shape

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
        
        # num_iterations = height * width  # Number of iterations

        h_blocks = math.ceil(height / self.p)
        w_blocks = math.ceil(width / self.p)
        
        h_pad = h_blocks * self.p
        w_pad = w_blocks * self.p
        
        num_iterations = h_pad * w_pad

        logistic_sequence = self.logistic_map(theta, initial_value, num_iterations - 1)
        # 混沌序列变型得到Q
        Q = self.reshape_sequence_to_Q(logistic_sequence, height, width)
        # 掩码矩阵Q置乱
        diffused_channels = self.channel_diffusion(channels, Q)
        # diffused_channels = channels # 消融实验
        
        # # 新逻辑：各通道掩码不同（提升直方图效果）
        # num_iterations = height * width * depth  # Number of iterations
        # logistic_sequence = self.logistic_map(theta, initial_value, num_iterations - 1)
        # # 混沌序列变型得到Q
        # Q = self.reshape_sequence_to_global_mask(logistic_sequence, plain_img.shape)
        # # 掩码矩阵Q置乱
        # diffused_channels = self.channel_diffusion(channels, Q)


        # ===== 阶段2：DNA加密 =====

        # ===== 阶段2.1：将混沌序列量化到8个DNA编码规则(已经在生成时完成) =====

        # ===== 阶段2.2：各通道转换为8位二进制字符串 =====
        
        channels_blocks = [self.split_into_blocks(ch) for ch in diffused_channels]

        bin_channels = [self.convert_to_8bit_binary(ch) for ch in channels_blocks]

        # ===== 阶段2.3：基于DNA编解码的加密 =====

        dna_channels = [self.binary_to_dna(bin_ch, x1) for bin_ch in bin_channels]
        bin_channels = [self.dna_to_binary(dna_ch, x2) for dna_ch in dna_channels]
        dna_channels = [self.binary_to_dna(bin_ch, x3) for bin_ch in bin_channels]
        bin_channels = [self.dna_to_binary(dna_ch, x4) for dna_ch in dna_channels]

        channels_blocks = [self.convert_binary_to_decimal(bin_ch) for bin_ch in bin_channels]

        channels = [self.reshape_blocks_to_channel(ch, height, width) for ch in channels_blocks]

        # ===== 阶段3：保存图像与生成密钥 =====

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
        height, width, depth = cipher_img.shape
        
        # 重新生成 Q 矩阵
        theta = 3.9999
        num_iterations = height * width  # Number of iterations
        
        h_blocks = math.ceil(height / self.p)
        w_blocks = math.ceil(width / self.p)
        
        h_pad = h_blocks * self.p
        w_pad = w_blocks * self.p
        
        num_iterations = h_pad * w_pad

        logistic_sequence = self.logistic_map(theta, initial_value_of_Q, num_iterations - 1)
        Q = self.reshape_sequence_to_Q(logistic_sequence, height, width)

        # theta = 3.9999
        # num_iterations = height * width * depth # Number of iterations
        # logistic_sequence = self.logistic_map(theta, initial_value_of_Q, num_iterations - 1)
        # # Q = self.reshape_sequence_to_Q(logistic_sequence, height, width)
        # Q = self.reshape_sequence_to_global_mask(logistic_sequence, cipher_img.shape)

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
        diffused_channels = [self.reshape_blocks_to_channel(ch, height, width) for ch in channels_blocks]


        # 解密Q置乱
        channels = self.channel_inverse_diffusion(diffused_channels, Q)
        # channels = diffused_channels

        decrypted_img = cv2.merge(channels)

        return decrypted_img


    def generate_with_retry(self, height, width):
        """
        封装了原代码中的同步性检查循环
        """
        if not self.enable_sync_check:
            return self.sequence_quantization(height, width)

        attempt_num = 5
        for i in range(attempt_num):
            m_seq, s_seq = self.sequence_quantization(height, width)
            # 简单的同步检查
            eq = [np.array_equal(m, s) for m, s in zip(m_seq, s_seq)]
            print(f"混沌同步结果检查：{eq}")
            if all(eq):
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


    def encry_and_decry_one_mat(self, plain_mat: np.ndarray, cipher_path: str, save_mode: int = 1)-> np.ndarray:
        """
        对单个矩阵进行加密解密,保存密文文件到磁盘.

        Args:
            plain_mat (np.ndarray): 明文矩阵(任意)
            cipher_path (str): 密文保存位置
            save_mode (int): 保存格式, 
                0 为原封不动的保存, 
                1 为仅保存前三个通道tiff格式(bgr)

        Returns:
            decrypted_mat (np.ndarray): 解密出的矩阵,预期与明文矩阵完全一致.
        
        """
        
        # 生成混沌序列
        height, width = plain_mat.shape[:2]
        master_seq, slave_seq = self.generate_with_retry(height, width)

        # self.visualize_chaos_seq(master_seq,
        #                         os.path.join(EXPERMENT_DIR, "draw_disp", f"x_int.png"),
        #                         128)
        # self.visualize_chaos_seq(slave_seq,
        #                         os.path.join(EXPERMENT_DIR, "draw_disp", f"y_int_{cnt}.png"),
        #                         128)
        # raise RuntimeError("debug here")

        # 加密
        cipher_img, key = self.encrypt(master_seq, plain_mat)
        if save_mode == 0:
            # 原状保存
            cv2.imwrite(cipher_path, cipher_img)
            print(f" 成功保存 {cipher_img.shape[2]} 通道图像至: {cipher_path}")
        elif save_mode == 1:
            # 保存为BGR彩色图像.tiff
            cv2.imwrite(cipher_path, cipher_img[:, :, :3])
            print(f" 成功保存 BGR 图像至: {cipher_path}")
        else:
            raise ValueError("save_mode 值错误,不存在这种保存模式")

        # 解密
        dec_img = self.decrypt(slave_seq, cipher_img, key)

        # 验证
        if not self.check_img_pixel(plain_mat, dec_img):
            print(f"Warning: Decrypted image does not match the original for {os.path.basename(cipher_path)}.")

        return dec_img


    def process_unimodal_folder(self, plain_imgs_dir, cipher_imgs_dir, decrypted_dir, k: int = 3):
        """
        批处理文件夹
        """
        os.makedirs(cipher_imgs_dir, exist_ok=True)
        os.makedirs(decrypted_dir, exist_ok=True)
        
        files = [f for f in os.listdir(plain_imgs_dir) if f.lower().endswith(('.tiff', '.tif'))]
        # print(f"type of files[0]: {type(files[0])}")
        # print(f"files[0]: \n{files[0]}")
        # raise RuntimeError("debug here")
        files.sort()
        cnt = 0
        for file_name in files:
            plain_path = os.path.join(plain_imgs_dir, file_name)
            cipher_path = os.path.join(cipher_imgs_dir, file_name)
            decrypted_path = os.path.join(decrypted_dir, file_name)
            cnt += 1
            if cnt > k:
                print(f"仅处理前{k}个项目")
                break

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

                self.visualize_chaos_seq(master_seq,
                                        os.path.join(EXPERMENT_DIR, "draw_disp", f"x_int_{cnt}.png"),
                                        128)
                # self.visualize_chaos_seq(slave_seq,
                #                         os.path.join(EXPERMENT_DIR, "draw_disp", f"y_int_{cnt}.png"),
                #                         128)
                # raise RuntimeError("debug here")

                # 加密
                cipher_img, key = self.encrypt(master_seq, plain_img)
                cv2.imwrite(cipher_path, cipher_img)
                print(f"成功保存 {cipher_img.shape[2]} 通道图像至: {cipher_path}")

                # 解密
                dec_img = self.decrypt(slave_seq, cipher_img, key)
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
    
    # 1. 实例化类
    encryptor = MultiEncryptor(
        model_path=MODEL_PATH, 
        password="my_secure_password",
        block_size=8,
        enable_sync_check=ENABLE_SYNC_CHECK
    )

    # 2. 调用批处理
    encryptor.process_unimodal_folder(
        plain_imgs_dir=PLAIN_IMGS_DIR,
        cipher_imgs_dir=CIPHER_IMGS_DIR,
        decrypted_dir=DECRYPTED_IMGS_DIR
    )
