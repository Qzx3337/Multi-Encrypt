import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 建议安装 tqdm 显示进度条: pip install tqdm
import csv

# ================= 配置区域 ================= #
# 对应 chaos.py 中的路径逻辑
BASE_EXPERIMENT_DIR = "experiments/w401/hyper_kvasir"
PLAIN_DIR = os.path.join(BASE_EXPERIMENT_DIR, "plain_img")
CIPHER_DIR = os.path.join(BASE_EXPERIMENT_DIR, "cipher_img")

# 新增：结果保存目录 (自动创建在同级)
ANALYSIS_OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "analysis")
PLOT_DIR = os.path.join(ANALYSIS_OUT_DIR, "correlation_plots")

# 图像格式
IMG_EXT = ".png"
# =========================================== #

class ImageAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(image_gray):
        """
        计算灰度图像的信息熵。
        理论最大值为 8.0。
        """
        # 计算直方图
        histogram, _ = np.histogram(image_gray.flatten(), bins=256, range=[0, 256])
        # 归一化得到概率
        probabilities = histogram / float(np.sum(histogram))
        # 过滤掉 0 概率，防止 log(0) 错误
        probabilities = probabilities[probabilities > 0]
        # 计算熵
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def get_pixel_pairs(image_gray, direction):
        """
        获取用于相关性分析的像素对 (x, y)。
        
        改进说明：
        使用 NumPy 切片代替循环，极大提高效率。
        修正了原代码只取主对角线的错误，现在取全图所有对角相邻点。
        """
        h, w = image_gray.shape
        
        if direction == 'horizontal':
            # (i, j) vs (i, j+1)
            x = image_gray[:, :-1].flatten()
            y = image_gray[:, 1:].flatten()
            
        elif direction == 'vertical':
            # (i, j) vs (i+1, j)
            x = image_gray[:-1, :].flatten()
            y = image_gray[1:, :].flatten()
            
        elif direction == 'diagonal':
            # (i, j) vs (i+1, j+1)
            # 改进：全矩阵切片，不再只是一条线
            x = image_gray[:-1, :-1].flatten()
            y = image_gray[1:, 1:].flatten()
            
        else:
            raise ValueError(f"Unknown direction: {direction}")
            
        return x, y

    @staticmethod
    def calculate_correlation_coefficient(x, y):
        """计算皮尔逊相关系数"""
        if len(x) == 0: return 0.0
        return np.corrcoef(x, y)[0, 1]

    def process_image(self, img_path, save_plot_prefix=None):
        """
        处理单张图片：计算熵、三个方向的相关系数，并可选绘图。
        """
        # 读取图片并转为灰度 (分析通常基于灰度或单通道)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}")
            return None
        
        # 如果是彩色图，转换为灰度进行统一分析，或者您可以修改此处只分析 R 通道
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 计算信息熵
        entropy = self.calculate_entropy(img_gray)

        # 2. 计算相关系数 & 绘图数据
        directions = ['horizontal', 'vertical', 'diagonal']
        results = {'entropy': entropy}

        # 准备绘图画布 (如果需要保存图)
        if save_plot_prefix:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Correlation Distribution: {os.path.basename(img_path)}', fontsize=16)

        for i, direction in enumerate(directions):
            x, y = self.get_pixel_pairs(img_gray, direction)
            corr = self.calculate_correlation_coefficient(x, y)
            results[f'corr_{direction}'] = corr

            if save_plot_prefix:
                ax = axes[i]
                # 绘制散点图
                # s=0.1, alpha=0.5 适合大量数据点，避免重叠成一团
                ax.scatter(x, y, s=0.5, c='blue', alpha=0.5)
                ax.set_title(f'{direction.capitalize()} (Corr: {corr:.4f})')
                ax.set_xlabel('Pixel (x)')
                ax.set_ylabel('Neighbor (y)')
                ax.set_xlim(0, 255)
                ax.set_ylim(0, 255)
                ax.set_aspect('equal')

        if save_plot_prefix:
            plt.tight_layout()
            plt.savefig(f"{save_plot_prefix}_corr.png", dpi=150)
            plt.close(fig)  # 必须关闭，否则内存泄漏

        return results

def batch_process_folder():
    # 1. 创建输出目录
    os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 2. 准备结果记录 CSV
    csv_path = os.path.join(ANALYSIS_OUT_DIR, "analysis_summary.csv")
    
    # 获取文件列表
    if not os.path.exists(CIPHER_DIR):
        print(f"Directory not found: {CIPHER_DIR}")
        return

    files = sorted([f for f in os.listdir(CIPHER_DIR) if f.lower().endswith(IMG_EXT)])
    
    print(f"Starting analysis on {len(files)} images...")
    print(f"Results will be saved to: {ANALYSIS_OUT_DIR}")

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'type', 'entropy', 'corr_horizontal', 'corr_vertical', 'corr_diagonal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 使用 tqdm 显示进度条
        for file_name in tqdm(files):
            # ============ 分析加密图 (Cipher) ============
            cipher_path = os.path.join(CIPHER_DIR, file_name)
            plot_save_path = os.path.join(PLOT_DIR, f"{os.path.splitext(file_name)[0]}_cipher")
            
            analyzer = ImageAnalyzer()
            
            # 计算并画图
            res_cipher = analyzer.process_image(cipher_path, save_plot_prefix=plot_save_path)
            
            if res_cipher:
                row = {'filename': file_name, 'type': 'cipher'}
                row.update(res_cipher)
                writer.writerow(row)

            # ============ 分析原图 (Plain) - 可选 ============
            # 如果需要对比，可以打开下面的注释。
            # 通常只需要数据对比，不需要每张原图都画一遍散点图（原图相关性高是常识）
            plain_path = os.path.join(PLAIN_DIR, file_name)
            if os.path.exists(plain_path):
                # 这里可选传入 save_plot_prefix，只计算数据不画图，节省时间
                plot_save_path = os.path.join(PLOT_DIR, f"{os.path.splitext(file_name)[0]}_plain")
                # res_plain = analyzer.process_image(plain_path, save_plot_prefix=None) 
                res_plain = analyzer.process_image(plain_path, save_plot_prefix=plot_save_path) 
                if res_plain:
                    row = {'filename': file_name, 'type': 'plain'}
                    row.update(res_plain)
                    writer.writerow(row)

    print("\nBatch analysis completed.")
    print(f"CSV Report: {csv_path}")
    print(f"Plots: {PLOT_DIR}")

if __name__ == "__main__":
    batch_process_folder()