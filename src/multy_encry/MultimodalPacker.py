import os
import cv2
import numpy as np
import pandas as pd
import struct
import glob

class MultimodalPacker:
    """
    多模态数据打包器
    负责将图像（N通道）与文本数据打包合并为（N+1）通道的矩阵，
    并支持反向解包。
    """
    def __init__(self):
        self.header_size = 4  # 使用4字节存储文本长度

    def _text_to_channel(self, text: str, height: int, width: int) -> np.ndarray:
        """
        内部方法：将文本转换为单通道矩阵
        策略：[4字节长度头] + [文本字节] + [随机噪声填充]
        """
        # 1. 文本编码为字节 (UTF-8)
        text_bytes = text.encode('utf-8')
        text_len = len(text_bytes)
        capacity = height * width
        max_payload = capacity - self.header_size

        # 2. 长度校验与截断
        if text_len > max_payload:
            # print(f"Warning: Text too long ({text_len} > {max_payload}), truncating.")
            text_bytes = text_bytes[:max_payload]
            text_len = max_payload

        # 3. 构建数据流
        # (1) 头部：4字节无符号整数存储有效文本长度
        header = struct.pack('I', text_len)
        
        # (2) 填充：计算剩余空间并生成随机噪声
        padding_len = capacity - self.header_size - text_len
        # 使用 numpy 生成 0-255 的随机噪声，类型为 uint8
        padding = np.random.randint(0, 256, padding_len, dtype=np.uint8)

        # (3) 拼接：Header + Text + Noise
        # 注意：frombuffer 需要字节流，所以 padding.tobytes()
        full_data_bytes = header + text_bytes + padding.tobytes()
        
        # 4. 转换为矩阵
        # frombuffer 将字节流转为一维 uint8 数组，然后 reshape
        channel_matrix = np.frombuffer(full_data_bytes, dtype=np.uint8).reshape((height, width))
        
        return channel_matrix

    def _channel_to_text(self, channel_matrix: np.ndarray) -> str:
        """
        内部方法：从单通道矩阵提取文本
        """
        try:
            # 1. 展平为一维数组
            flat_data = channel_matrix.flatten()
            
            # 2. 读取头部 (前4字节)
            header_bytes = flat_data[:self.header_size].tobytes()
            text_len = struct.unpack('I', header_bytes)[0]
            
            # 安全检查：防止读取出的长度超出数组范围（虽然理论上不会，但防脏数据）
            if text_len > flat_data.size - self.header_size:
                text_len = flat_data.size - self.header_size
            
            # 3. 提取有效载荷
            text_bytes = flat_data[self.header_size : self.header_size + text_len].tobytes()
            
            # 4. 解码
            return text_bytes.decode('utf-8', errors='ignore') # ignore 忽略截断导致的末尾乱码
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def pack(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        打包接口：将任意通道图像与文本合并。
        Args:
            image: 输入图像矩阵 (H, W, C) 或 (H, W)
            text: 待嵌入文本
        Returns:
            combined_img: (H, W, C+1) 矩阵
        """
        # 获取图像尺寸
        if len(image.shape) == 2:
            height, width = image.shape
            channels = [image] # 单通道图处理为列表
        else:
            height, width = image.shape[:2]
            # 动态分离通道：无论多少个通道，cv2.split 都能正确拆分
            channels = list(cv2.split(image))
        
        # 生成文本通道
        text_channel = self._text_to_channel(text, height, width)
        
        # 合并通道：原通道列表 + 文本通道
        channels.append(text_channel)
        combined_img = cv2.merge(channels)
        
        return combined_img

    def unpack(self, combined_img: np.ndarray):
        """
        解包接口：分离图像和文本。
        Args:
            combined_img: (H, W, N) 矩阵
        Returns:
            image: (H, W, N-1) 图像矩阵
            text: 提取的字符串
        """
        if len(combined_img.shape) < 3:
            raise ValueError("Input image must have at least one channel + text channel.")

        # 动态分离通道
        all_channels = cv2.split(combined_img)
        
        # 策略：前 N-1 个通道是图像，最后一个通道是文本
        image_channels = all_channels[:-1]
        text_channel = all_channels[-1]
        
        # 还原图像
        if len(image_channels) == 1:
            image = image_channels[0]
        else:
            image = cv2.merge(image_channels)
            
        # 还原文本
        text = self._channel_to_text(text_channel)
        
        return image, text

# --- 以下是测试代码，用于验证类的可行性 ---

def run_feasibility_test(csv_path, img_dir, output_dir, k=5):
    """
    批量可行性测试函数
    """
    print(f"--- Starting Feasibility Test (Top {k} items) ---")
    
    # 1. 准备环境
    images_output_subdir = os.path.join(output_dir, "images")
    os.makedirs(images_output_subdir, exist_ok=True)
    results_list = []
    packer = MultimodalPacker()
    
    # 2. 读取数据索引
    # 假设CSV没有表头或第一列有效，这里按行读取
    # 也可以用 pandas: df = pd.read_csv(csv_path)
    try:
        df = pd.read_csv(csv_path, header=0) # header=0表示第一行为表头，根据实际情况调整
        # iloc[:, 3:] 提取所有行和第四列开始到最后一列
        # fillna('') 将表格的空位置代换为"" 不然会出现 nan
        # apply(lambda x: ', '.join(x), axis=1) 连缀所有列的时候, 使用', ' 分割
        texts = df.iloc[:, 3:].fillna('').astype(str).apply(lambda x: ', '.join(x), axis=1).tolist()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 获取图片列表并排序，确保与CSV行对应
    img_files = glob.glob(os.path.join(img_dir, '*.tiff'))
    img_files.sort()

    # 校验数量
    num_items = min(len(texts), len(img_files), k)
    print(f"Found {len(texts)} texts and {len(img_files)} images. Processing first {num_items} items.")

    # 3. 循环处理
    for i in range(num_items):
        print(f"\n[Item {i}] Processing...")
        
        # A. 读取原始数据
        original_text = texts[i]
        img_path = img_files[i]
        base_name = os.path.basename(img_path) # 获取文件名，如 "001.tiff"
        original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # UNCHANGED 以保留原图所有通道(如Alpha)
        
        if original_img is None:
            print(f"  Error: Could not read image {img_path}")
            continue
            
        print(f"  Original Image Shape: {original_img.shape}")
        print(f"  Original Text Len: {len(original_text)}")

        # B. 打包 (Pack)
        # 此时 packed_matrix 是在内存中的 (H, W, C+1) 矩阵
        packed_matrix = packer.pack(original_img, original_text)
        print(f"  Packed Matrix Shape: {packed_matrix.shape} (Channel count increased by 1)")
        
        # --- 模拟：这里是 main.py 中传递给 Encryptor 的地方 ---
        # encryptor.encrypt(packed_matrix) ...
        # cipher = ...
        # decrypted_matrix = encryptor.decrypt(cipher) ...
        # 假设解密完美还原，直接将 packed_matrix 传给解包逻辑
        decrypted_matrix_simulation = packed_matrix 
        # ---------------------------------------------------

        # C. 解包 (Unpack)
        restored_img, restored_text = packer.unpack(decrypted_matrix_simulation)
        
        # D. 验证与保存
        # 验证文本
        if original_text == restored_text:
            print("  [SUCCESS] Text recovered perfectly.")
        else:
            # 如果发生了截断，这里会不等
            if len(restored_text) < len(original_text):
                 print(f"  [INFO] Text was truncated (expected behavior if too long).")
            else:
                 print(f"  [FAIL] Text mismatch!")
        
        # 验证图像形状
        if original_img.shape == restored_img.shape:
             print("  [SUCCESS] Image shape matches.")
        else:
             print("  [FAIL] Image shape mismatch.")

        # 1. 保存图片到 images 子目录
        # 保持原文件名，不加 "restored_" 前缀，或者根据需求保留
        save_img_path = os.path.join(images_output_subdir, base_name)
        cv2.imwrite(save_img_path, restored_img)
        print(f"  Saved image to: {save_img_path}")
        
        # 2. 收集文本信息到内存列表 (不直接写文件)
        results_list.append({
            "filename": base_name,      # 记录对应的图片文件名作为索引
            "restored_text": restored_text
        })

    # 4. 循环结束后：统一保存 CSV
    if results_list:
        csv_output_path = os.path.join(output_dir, "message.csv")
        
        # 使用 Pandas 创建 DataFrame 并保存
        result_df = pd.DataFrame(results_list)
        
        # encoding='utf-8-sig': 关键！
        # 加上 sig 签名可以确保 Excel 在 Windows 上打开中文不乱码
        result_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[SUCCESS] All text data saved to: {csv_output_path}")
    
    print("--- Feasibility Test Completed ---")
            
    print("\n--- Feasibility Test Completed ---")

# 使用示例 (请根据实际路径修改)
if __name__ == "__main__":
    # 配置路径
    # experiments\w501\zky_plain\images
    # experiments\w501\zky_plain\messages.csv
    CSV_PATH = os.path.join("experiments", "w501", "zky_plain", "messages.csv")
    IMG_DIR = os.path.join("experiments", "w501", "zky_plain", "images")
    OUTPUT_DIR = os.path.join("experiments", "w501", "zky_decry")
    K = 3 # 只测试前3个
    
    # 只有当路径存在时才运行，防止报错
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        run_feasibility_test(CSV_PATH, IMG_DIR, OUTPUT_DIR, K)
    else:
        print("Please configure input paths in __main__ to run the test.")

        