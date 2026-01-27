import os
import cv2
import numpy as np  # 【修改点】：引入 numpy 进行统计计算
import matplotlib.pyplot as plt

def calculate_limits(img, is_log):
    """
    【新增功能】：计算图像直方图数据的 mu +/- 5*sigma 范围。
    """
    # 1. 计算所有通道的直方图数据
    hist_data = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist_data.append(hist.flatten())
    
    # 合并三个通道的数据为一个大数组进行统计
    all_data = np.concatenate(hist_data)
    
    if is_log:
        # --- Log 模式统计逻辑 ---
        # 避免 log(0)，加一个极小值 epsilon (例如 1e-6 或 1)
        # 这里加 1 是因为像素数通常是整数，log(1)=0，不会产生负无穷
        log_data = np.log(all_data + 1)
        
        mu = np.mean(log_data)
        sigma = np.std(log_data)
        
        # 计算 Log 域下的上下限
        upper_log = mu + 5 * sigma
        lower_log = mu - 5 * sigma
        
        # 【关键】：因为 plt.yscale('log') 还是使用线性数值标记刻度，
        # 所以我们需要把 log 域的统计值还原回线性域 (exp)
        y_max = np.exp(upper_log)
        y_min = np.exp(lower_log)
        
        # Log 坐标下，下限不能小于等于 0，设置一个安全下限（如 1）
        if y_min < 1: 
            y_min = 1
            
    else:
        # --- 线性模式统计逻辑 ---
        mu = np.mean(all_data)
        sigma = np.std(all_data)
        
        y_max = mu + 3 * sigma
        y_min = mu - 3 * sigma
        
        # 线性坐标下，像素数量不可能为负，截断到 0
        if y_min < 0:
            y_min = 0
            
    return y_min, y_max

def plot_and_save_histogram(img, save_path, log_scale=False, y_limits=None):
    """
    绘制并保存直方图。
    【修改点】：增加了 y_limits 参数用于固定纵坐标范围。
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
    
    # 【修改点】：应用计算好的纵坐标范围
    if y_limits is not None:
        plt.ylim(y_limits)

    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist_data = hist.flatten()
        plt.plot(hist_data, color=color, linewidth=1)
        # 调整透明度为 0.1 避免遮挡
        plt.fill_between(range(256), hist_data, color=color, alpha=0.1)

    plt.savefig(save_path)
    plt.close()

def process_paired_folders(plain_dir, cipher_dir, 
                           plain_out_dict, cipher_out_dict):
    """
    【核心修改】：成对处理文件夹。
    同时读取 Plain 和 Cipher 图片，计算共同的纵坐标范围，然后绘图。
    """
    # 1. 创建所有输出目录
    for d in list(plain_out_dict.values()) + list(cipher_out_dict.values()):
        if d and not os.path.exists(d):
            os.makedirs(d)

    # 2. 以 Plain 文件夹的文件列表为基准
    if not os.path.exists(plain_dir):
        print(f"Error: Source directory {plain_dir} does not exist.")
        return

    files = os.listdir(plain_dir)
    count = 0
    
    print(f"Processing paired histograms for: {plain_dir} AND {cipher_dir} ...")
    
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            
            # --- 步骤 A: 读取成对的图片 ---
            plain_path = os.path.join(plain_dir, filename)
            cipher_path = os.path.join(cipher_dir, filename)
            
            img_plain = cv2.imread(plain_path)
            
            # 如果加密文件夹里没有对应的图，或者读取失败，则跳过对比逻辑（或者你可以选择单独处理）
            if not os.path.exists(cipher_path):
                print(f"Warning: Cipher counterpart not found for {filename}")
                continue
                
            img_cipher = cv2.imread(cipher_path)

            if img_plain is None or img_cipher is None:
                continue

            output_filename = f"{os.path.splitext(filename)[0]}_hist.png"
            
            try:
                # --- 步骤 B: 确定 线性坐标 的共同范围 ---
                # 分别计算两个图的建议范围 (mu +/- 5sigma)
                p_lin_min, p_lin_max = calculate_limits(img_plain, is_log=False)
                c_lin_min, c_lin_max = calculate_limits(img_cipher, is_log=False)
                
                # 【核心逻辑】：取并集。
                # 为了对比，Y轴上限必须能容纳两者中最高的那个上限。
                # 通常 Plain 图会有很高的峰值，而 Cipher 图很平。如果不统一，Cipher 图会被放大看噪声。
                # 统一后，Cipher 图应该在下方显示为一条平缓的线。
                shared_linear_max = max(p_lin_max, c_lin_max)
                shared_linear_min = min(p_lin_min, c_lin_min) # 通常是 0
                linear_limits = [shared_linear_min, shared_linear_max]

                # --- 步骤 C: 确定 Log 坐标 的共同范围 ---
                p_log_min, p_log_max = calculate_limits(img_plain, is_log=True)
                c_log_min, c_log_max = calculate_limits(img_cipher, is_log=True)
                
                shared_log_max = max(p_log_max, c_log_max)
                shared_log_min = min(p_log_min, c_log_min)
                log_limits = [shared_log_min, shared_log_max]

                # --- 步骤 D: 绘图并保存 ---
                
                # 1. Plain Image Plots
                if plain_out_dict.get("linear"):
                    save_path = os.path.join(plain_out_dict["linear"], output_filename)
                    plot_and_save_histogram(img_plain, save_path, False, linear_limits)
                
                if plain_out_dict.get("log"):
                    save_path = os.path.join(plain_out_dict["log"], output_filename)
                    plot_and_save_histogram(img_plain, save_path, True, log_limits)

                # 2. Cipher Image Plots (使用相同的 limits)
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
                import traceback
                traceback.print_exc()

    print(f"Finished. Processed {count} image pairs.\n")

if __name__ == "__main__":
    # --- 配置路径 ---
    
    plain_dir = "pictures/data01/plain_img"
    cipher_dir = "pictures/data01/cipher_img"
    
    # 定义输出路径字典
    plain_out = {
        "linear": "pictures/data01/plain_hist",
        "log": "pictures/data01/plain_hist_log"
    }
    
    cipher_out = {
        "linear": "pictures/data01/cipher_hist",
        "log": "pictures/data01/cipher_hist_log"
    }

    # 执行成对处理
    process_paired_folders(plain_dir, cipher_dir, plain_out, cipher_out)

