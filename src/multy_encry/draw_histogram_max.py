import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_max_pixel_count(img):
    """
    【新增简单逻辑】：
    不计算均值方差，直接遍历RGB三个通道，
    返回这张图片中出现次数最多的那个像素值的频次（即直方图的最高峰）。
    """
    max_val = 0
    for i in range(3):
        # 计算该通道直方图
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        # 获取该通道最大值
        current_max = np.max(hist)
        if current_max > max_val:
            max_val = current_max
    return max_val

def plot_and_save_histogram(img, save_path, log_scale=False, y_limits=None):
    """
    绘制并保存直方图（保持不变）。
    """
    plt.figure()
    
    if log_scale:
        plt.title("RGB Histogram (Log Scale)")
        plt.yscale('log')
    else:
        plt.title("RGB Histogram (Linear Scale)")

    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.xlim([0, 256])
    
    # 应用传入的纵坐标范围
    if y_limits is not None:
        plt.ylim(y_limits)

    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist_data = hist.flatten()
        plt.plot(hist_data, color=color, linewidth=1)
        plt.fill_between(range(256), hist_data, color=color, alpha=0.1)

    plt.savefig(save_path)
    plt.close()

def process_paired_folders(plain_dir, cipher_dir, 
                           plain_out_dict, cipher_out_dict):
    """
    成对处理文件夹，使用【最大值覆盖】策略确定坐标轴。
    """
    # 1. 创建所有输出目录
    for d in list(plain_out_dict.values()) + list(cipher_out_dict.values()):
        if d and not os.path.exists(d):
            os.makedirs(d)

    # 2. 检查源目录
    if not os.path.exists(plain_dir):
        print(f"Error: Source directory {plain_dir} does not exist.")
        return

    files = os.listdir(plain_dir)
    count = 0
    
    print(f"Processing (MAX Strategy) for: {plain_dir} AND {cipher_dir} ...")
    
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            
            plain_path = os.path.join(plain_dir, filename)
            cipher_path = os.path.join(cipher_dir, filename)
            
            img_plain = cv2.imread(plain_path)
            
            if not os.path.exists(cipher_path):
                print(f"Warning: Cipher counterpart not found for {filename}")
                continue
                
            img_cipher = cv2.imread(cipher_path)

            if img_plain is None or img_cipher is None:
                continue

            output_filename = f"{os.path.splitext(filename)[0]}_hist.png"
            
            try:
                # --- 【核心修改区域 Start】 ---
                
                # 1. 获取两张图各自的最高峰值
                max_p = get_max_pixel_count(img_plain)
                max_c = get_max_pixel_count(img_cipher)
                
                # 2. 取两者的最大值作为基准
                global_max = max(max_p, max_c)
                
                # 3. 设置上界：在最大值基础上增加 10% 的留白 (乘以 1.1)
                y_top = global_max * 1.1
                
                # 防止全黑图片导致 max 为 0
                if y_top == 0: 
                    y_top = 10 

                # 4. 定义范围
                # 线性坐标：下界固定为 0
                linear_limits = [0, y_top]
                
                # 对数坐标：下界不能为0（log(0)无定义）。
                # 设置为 0.5 或 1，保证 Y 轴起始点清晰。
                # 既然上限也是由像素数量决定的，这里直接复用 y_top 即可。
                log_limits = [0.5, y_top] 

                # --- 【核心修改区域 End】 ---

                # 绘图逻辑（与之前一致）
                
                # 1. Plain Image Plots
                if plain_out_dict.get("linear"):
                    save_path = os.path.join(plain_out_dict["linear"], output_filename)
                    plot_and_save_histogram(img_plain, save_path, False, linear_limits)
                
                if plain_out_dict.get("log"):
                    save_path = os.path.join(plain_out_dict["log"], output_filename)
                    plot_and_save_histogram(img_plain, save_path, True, log_limits)

                # 2. Cipher Image Plots
                if cipher_out_dict.get("linear"):
                    save_path = os.path.join(cipher_out_dict["linear"], output_filename)
                    plot_and_save_histogram(img_cipher, save_path, False, linear_limits)
                
                if cipher_out_dict.get("log"):
                    save_path = os.path.join(cipher_out_dict["log"], output_filename)
                    plot_and_save_histogram(img_cipher, save_path, True, log_limits)
                
                count += 1
                if count % 10 == 0:
                    print(f"Processed pairs: {count}...")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"Finished. Processed {count} image pairs.\n")

if __name__ == "__main__":
    # --- 配置路径 ---
    
    plain_dir = "pictures/data01/plain_img"
    cipher_dir = "pictures/data01/cipher_img"
    
    plain_out = {
        "linear": "pictures/data01/plain_hist",
        "log": "pictures/data01/plain_hist_log"
    }
    
    cipher_out = {
        "linear": "pictures/data01/cipher_hist",
        "log": "pictures/data01/cipher_hist_log"
    }

    process_paired_folders(plain_dir, cipher_dir, plain_out, cipher_out)

    