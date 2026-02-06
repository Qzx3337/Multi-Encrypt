import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# experiments\w502\zky_plain\images
PLAIN_DIR= os.path.join("experiments", "w502", "zky_plain", "images")
CIPHER_DIR = os.path.join("experiments", "w502", "zky_cipher")
HIST_DIR = os.path.join("experiments", "w502", "zky_hist")

def plot_histogram_comparison(plain_path, cipher_path, hist_path, log_scale=True):
    """
    绘制明文和密文图像在B, G, R三个通道上的直方图对比。
    
    Args:
        plain_path (str): 明文图像路径
        cipher_path (str): 密文图像路径
        hist_path (str): 结果保存路径 (包含文件名，如 result/test_hist.png)
        log_scale (bool): 是否开启对数坐标
    """
    # 1. 读取图像
    img_plain = cv2.imread(plain_path)
    img_cipher = cv2.imread(cipher_path)

    if img_plain is None:
        print(f"Error: 无法读取明文图像: {plain_path}")
        return
    if img_cipher is None:
        print(f"Error: 无法读取密文图像: {cipher_path}")
        return

    # 2. 设置绘图风格
    # 使用浅一点的颜色以便论文排版 (B, G, R 顺序对应的浅色)
    colors = ['#5c9dff', '#6bdc6b', '#ff6b6b']  # 浅蓝, 浅绿, 浅红
    channel_names = ['Blue Channel', 'Green Channel', 'Red Channel']
    
    # 创建画布：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=100)
    # 调整子图间距
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # 3. 循环处理 B, G, R 三个通道
    for i in range(3):
        # 获取当前通道数据 (OpenCV默认为BGR顺序: 0-B, 1-G, 2-R)
        # 即使通道数>3，也只取前三个
        plain_chan = img_plain[:, :, i]
        cipher_chan = img_cipher[:, :, i]

        # 计算直方图
        hist_plain = cv2.calcHist([plain_chan], [0], None, [256], [0, 256])
        hist_cipher = cv2.calcHist([cipher_chan], [0], None, [256], [0, 256])

        # 获取当前通道的绘图轴
        ax_plain = axes[0, i]
        ax_cipher = axes[1, i]

        # --- 绘制明文直方图 ---
        ax_plain.fill_between(range(256), hist_plain.flatten(), color=colors[i], alpha=0.6)
        ax_plain.set_xlim([0, 255])
        ax_plain.set_title(f'Plain - {channel_names[i]}')
        # ax_plain.grid(True, linestyle='--', alpha=0.3)

        # --- 绘制密文直方图 ---
        ax_cipher.fill_between(range(256), hist_cipher.flatten(), color=colors[i], alpha=0.6)
        ax_cipher.set_xlim([0, 255])
        ax_cipher.set_title(f'Cipher - {channel_names[i]}')
        # ax_cipher.grid(True, linestyle='--', alpha=0.3)

        # --- 处理对数坐标与轴对齐逻辑 ---
        if log_scale:
            # 开启对数坐标
            ax_plain.set_yscale('log')
            ax_cipher.set_yscale('log')
            
            # 计算两者的最大值，以便统一Y轴范围
            # 注意：加1是为了防止log(0)错误，虽然fill_between通常处理得很好
            max_val = max(hist_plain.max(), hist_cipher.max())
            # 设置下限为1 (log坐标下不能为0)，上限稍微留点余量
            y_limit = [0.5, max_val * 2] # *2 是为了给上方留点空间
            
            # 强制对齐两个图的Y轴
            ax_plain.set_ylim(y_limit)
            ax_cipher.set_ylim(y_limit)
        else:
            # 线性坐标下，通常直方图从0开始
            ax_plain.set_ylim(bottom=0)
            ax_cipher.set_ylim(bottom=0)

    # 4. 保存图像
    # 确保目录存在
    save_dir = os.path.dirname(hist_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close() # 关闭画布释放内存
    print(f"Saved histogram comparison to: {hist_path}")


def batch_plot_histograms(plain_dir, cipher_dir, hist_dir, log_scale=True):
    """
    批量处理文件夹下的图像直方图对比。
    
    Args:
        plain_dir (str): 明文图像文件夹路径
        cipher_dir (str): 密文图像文件夹路径
        hist_dir (str): 结果保存文件夹路径
        log_scale (bool): 是否开启对数坐标
    """
    # 确保输出目录存在
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)

    # 获取明文文件夹下的所有文件
    files = os.listdir(plain_dir)
    # 支持的图像扩展名
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    count = 0
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_exts:
            continue

        # 构建完整路径
        p_path = os.path.join(plain_dir, filename)
        c_path = os.path.join(cipher_dir, filename)
        
        # 构建输出文件名：原文件名 + _hist + .png
        h_path = os.path.join(hist_dir, f"{name}_hist.png")

        # 检查对应的密文文件是否存在
        if not os.path.exists(c_path):
            print(f"Warning: 找不到对应的密文图像 {filename}，跳过。")
            continue

        # 调用单次绘图函数
        plot_histogram_comparison(p_path, c_path, h_path, log_scale)
        count += 1

    print(f"--- 批量处理完成，共生成 {count} 张直方图对比图 ---")


# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据你的实际路径修改下面的变量
    
    # 示例 1: 单张测试
    # plot_histogram_comparison(
    #     plain_path='data/plain/lena.png', 
    #     cipher_path='data/cipher/lena.png', 
    #     hist_path='results/lena_hist_compare.png', 
    #     log_scale=True
    # )

    # 示例 2: 批量测试
    batch_plot_histograms(
        plain_dir=PLAIN_DIR,
        cipher_dir=CIPHER_DIR,
        hist_dir=HIST_DIR,
        log_scale=True
    )

    pass

