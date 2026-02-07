import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DATABASE_NAME = "hyper_kvasir" # 更换数据集
EXP_DATA_DIR = os.path.join("experiments", "w502", DATABASE_NAME) # 更换实验
ENABLE_LOG_SCALE = True # 选择是否开启 log 坐标

# 路径配置
PLAIN_DIR= os.path.join("data", DATABASE_NAME, "images")
CIPHER_DIR = os.path.join(EXP_DATA_DIR, "cipher")
HIST_DIR = os.path.join(EXP_DATA_DIR, "hist")

def plot_histogram_comparison(plain_path, cipher_path, hist_path, log_scale=True):
    """
    绘制布局为 2行4列 的对比图：
    第一列：明文图像 vs 密文图像
    后三列：B, G, R 通道的直方图
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
    colors = ['#5c9dff', '#6bdc6b', '#ff6b6b']  # 浅蓝, 浅绿, 浅红
    
    # 创建画布：2行4列 (宽度增加以容纳原图)
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), dpi=100)
    
    # 自动调整布局，减少留白
    plt.tight_layout()
    # 也可以手动微调边距 (left, right, top, bottom)
    plt.subplots_adjust(wspace=0.25, hspace=0.15)

    # --- 第一列：绘制原图 (Row 0, Col 0) 与 密文图 (Row 1, Col 0) ---
    
    # OpenCV读取的是BGR，matplotlib显示需要RGB，需要转换
    img_plain_rgb = cv2.cvtColor(img_plain, cv2.COLOR_BGR2RGB)
    img_cipher_rgb = cv2.cvtColor(img_cipher, cv2.COLOR_BGR2RGB)

    # 绘制明文图像
    axes[0, 0].imshow(img_plain_rgb)
    axes[0, 0].axis('off') # 关闭坐标轴刻度，看起来更整洁

    # 绘制密文图像
    axes[1, 0].imshow(img_cipher_rgb)
    axes[1, 0].axis('off')

    # --- 后三列：绘制 B, G, R 直方图 ---
    for i in range(3):
        # 获取当前通道数据
        plain_chan = img_plain[:, :, i]
        cipher_chan = img_cipher[:, :, i]

        # 计算直方图
        hist_plain = cv2.calcHist([plain_chan], [0], None, [256], [0, 256])
        hist_cipher = cv2.calcHist([cipher_chan], [0], None, [256], [0, 256])

        # 获取绘图轴 (注意：列索引要 +1，因为第0列是图像)
        ax_plain = axes[0, 3 - i]
        ax_cipher = axes[1, 3 - i]

        # --- 绘制明文直方图 ---
        ax_plain.fill_between(range(256), hist_plain.flatten(), color=colors[i], alpha=0.6)
        ax_plain.set_xlim([0, 255])
        # ax_plain.set_xticks([]) # 如果想连X轴数字也不要，可以取消注释这行

        # --- 绘制密文直方图 ---
        ax_cipher.fill_between(range(256), hist_cipher.flatten(), color=colors[i], alpha=0.6)
        ax_cipher.set_xlim([0, 255])

        # --- 处理对数坐标与轴对齐逻辑 ---
        if log_scale:
            ax_plain.set_yscale('log')
            ax_cipher.set_yscale('log')
            
            # 统一 Y 轴范围
            max_val = max(hist_plain.max(), hist_cipher.max())
            y_limit = [0.5, max_val * 2]
            
            ax_plain.set_ylim(y_limit)
            ax_cipher.set_ylim(y_limit)
        else:
            ax_plain.set_ylim(bottom=0)
            ax_cipher.set_ylim(bottom=0)

    # 4. 保存图像
    save_dir = os.path.dirname(hist_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(hist_path, bbox_inches='tight', pad_inches=0.1) # pad_inches减小边缘空白
    plt.close()
    print(f"Saved histogram comparison to: {hist_path}")


def batch_plot_histograms(plain_dir, cipher_dir, hist_dir, log_scale=True):
    """
    批量处理文件夹下的图像直方图对比。
    """
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)

    files = os.listdir(plain_dir)
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    count = 0
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_exts:
            continue

        p_path = os.path.join(plain_dir, filename)
        c_path = os.path.join(cipher_dir, filename)
        
        # 结果文件名
        h_path = os.path.join(hist_dir, f"{name}_hist.png")

        if not os.path.exists(c_path):
            print(f"Warning: 找不到对应的密文图像 {filename}，跳过。")
            continue

        plot_histogram_comparison(p_path, c_path, h_path, log_scale)
        count += 1

    print(f"--- 批量处理完成，共生成 {count} 张直方图对比图 ---")


if __name__ == "__main__":
    # 执行批量处理
    batch_plot_histograms(
        plain_dir=PLAIN_DIR,
        cipher_dir=CIPHER_DIR,
        hist_dir=HIST_DIR,
        log_scale=ENABLE_LOG_SCALE
    )
    