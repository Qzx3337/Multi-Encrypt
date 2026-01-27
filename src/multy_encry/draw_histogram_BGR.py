
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --- 配置路径 (集中管理) ---

# 1. 定义实验的主根目录 
BASE_EXPERIMENT_DIR = "experiments/w401/hyper_kvasir"

# 2. 定义输入目录 
PLAIN_DIR = os.path.join(BASE_EXPERIMENT_DIR, "plain_img")   # 原图文件夹
CIPHER_DIR = os.path.join(BASE_EXPERIMENT_DIR, "cipher_img") # 密图文件夹

# 3. 定义分析结果的总输出目录 
ANALYSIS_DIR = os.path.join(BASE_EXPERIMENT_DIR, "analysis")

# --- ------------------ ---

def draw_and_save_hist(data, save_path, channel_color, use_log, y_limit=None, title_suffix=""):
    """
    绘制并保存单个通道的直方图
    """
    plt.figure(figsize=(6, 4))
    
    # 绘制直方图
    # bins=256, range=[0, 256] 确保覆盖所有像素值
    plt.hist(data.ravel(), bins=256, range=[0, 256], color=channel_color, alpha=0.99)
    
    # 设置坐标轴
    plt.xlim([0, 256])
    
    if use_log:
        plt.yscale('log')
        plt.ylabel("Frequency (Log Scale)")
        # Log模式下，如果要求对齐，则手动设置ylim
        if y_limit:
            # log坐标下底不能为0，通常设为0.5或1，顶设为计算出的最大值*1.1以留白
            plt.ylim(bottom=0.5, top=y_limit * 1.5) 
    else:
        plt.ylabel("Frequency")
        # 线性模式下，通常自动调整，但如果传入了limit也可以设置
        # 你的需求是：线性模式下不强制对齐，所以这里通常忽略 y_limit
        pass

    plt.title(f"Histogram {title_suffix}")
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # 保存并关闭，释放内存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_comparison_histograms(plain_dir, cipher_dir, output_root="RGB_hist_log", use_log=True):
    """
    主处理函数
    Args:
        plain_dir(str): 明文图片目录
        cipher_dir(str): 密文图片目录
        output_root(str): 输出根目录
        use_log(bool): 是否使用对数坐标 (默认 True)
    """
    # 1. 创建输出目录
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")

    # 获取所有png文件 (假设明文和密文文件名一致)
    plain_files = sorted(glob(os.path.join(plain_dir, "*.png")))
    
    print(f"Found {len(plain_files)} images to process...")
    print(f"Mode: {'Log Scale (Aligned)' if use_log else 'Linear Scale (Auto)'}")

    for p_path in plain_files:
        filename = os.path.basename(p_path)
        c_path = os.path.join(cipher_dir, filename)
        
        # 检查对应的密文文件是否存在
        if not os.path.exists(c_path):
            print(f"Warning: Cipher image for {filename} not found. Skipping.")
            continue
            
        # 读取图片 (OpenCV读取默认为BGR)
        img_plain = cv2.imread(p_path)
        img_cipher = cv2.imread(c_path)
        
        if img_plain is None or img_cipher is None:
            print(f"Error reading {filename}. Skipping.")
            continue

        # 分离通道 (OpenCV是BGR，转为RGB顺序方便处理)
        # 0:B, 1:G, 2:R -> 我们统一按 R, G, B 处理
        planes_p = {'R': img_plain[:,:,2], 'G': img_plain[:,:,1], 'B': img_plain[:,:,0]}
        planes_c = {'R': img_cipher[:,:,2], 'G': img_cipher[:,:,1], 'B': img_cipher[:,:,0]}
        
        # 定义颜色映射，用于画图颜色
        colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
        
        # 获取文件名主体 (去掉 .png)
        name_base = os.path.splitext(filename)[0]

        # 遍历三个通道 R, G, B
        for channel in ['R', 'G', 'B']:
            data_p = planes_p[channel]
            data_c = planes_c[channel]
            
            # === 核心逻辑：计算对齐所需的 Y 轴上限 ===
            current_y_limit = None
            if use_log:
                # 预先计算直方图频数，找出两张图中最大的频数
                # np.histogram 返回 (counts, bin_edges)
                hist_p, _ = np.histogram(data_p.ravel(), 256, [0, 256])
                hist_c, _ = np.histogram(data_c.ravel(), 256, [0, 256])
                
                max_freq_p = hist_p.max()
                max_freq_c = hist_c.max()
                
                # 取二者最大值，作为统一的 Y 轴上限
                current_y_limit = max(max_freq_p, max_freq_c)
            
            # === 保存路径 ===
            # 明文输出名: Plain_原文件名_R.png
            save_name_p = f"{name_base}_plain_{channel}.png"
            save_path_p = os.path.join(output_root, save_name_p)
            
            # 密文输出名: Cipher_原文件名_R.png
            save_name_c = f"{name_base}_cipher_{channel}.png"
            save_path_c = os.path.join(output_root, save_name_c)
            
            # === 绘图 ===
            # 绘制明文
            draw_and_save_hist(
                data_p, save_path_p, colors[channel], 
                use_log, current_y_limit, 
                title_suffix=f"(Plain - {channel})"
            )
            
            # 绘制密文
            draw_and_save_hist(
                data_c, save_path_c, colors[channel], 
                use_log, current_y_limit, 
                title_suffix=f"(Cipher - {channel})"
            )
            
        print(f"Processed {filename} -> {output_root}/...")

    print("All Done.")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 请修改为你的实际路径
    p_dir = PLAIN_DIR
    c_dir = CIPHER_DIR  # 密文文件夹路径
    output_root = os.path.join(ANALYSIS_DIR, "RGB_hist_log")
    
    # 开关控制：
    # use_log=True  -> 对数坐标，且明文密文Y轴强制对齐 (推荐)
    # use_log=False -> 线性坐标，Y轴自动缩放不对齐
    generate_comparison_histograms(p_dir, c_dir, output_root, use_log=True)

