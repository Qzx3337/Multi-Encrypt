import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random

DATABASE_NAME = "test" 
EXP_NAME = "w504" 

# --- 路径配置 ---
PLAIN_DIR = os.path.join("data", DATABASE_NAME, "images")
CIPHER_DIR = os.path.join("experiments", EXP_NAME, DATABASE_NAME, "cipher")
RESULT_DIR = os.path.join("experiments", EXP_NAME, DATABASE_NAME, "correlation_analysis")
CSV_PATH = os.path.join(RESULT_DIR, "correlation_results.csv")

N_SAMPLES = 9000

def get_adjacent_pixels(gray_img, direction, n_samples=3000):
    """随机获取图像中相邻像素对的值 (x, y)。"""
    h, w = gray_img.shape
    x = []
    y = []
    
    max_attempts = n_samples * 5
    count = 0
    attempts = 0
    
    while count < n_samples and attempts < max_attempts:
        row = np.random.randint(0, h - 1)
        col = np.random.randint(0, w - 1)
        
        val_x = gray_img[row, col]
        val_y = 0
        
        if direction == 'Horizontal':
            val_y = gray_img[row, col + 1]
        elif direction == 'Vertical':
            val_y = gray_img[row + 1, col]
        elif direction == 'Diagonal':
            val_y = gray_img[row + 1, col + 1]
            
        x.append(val_x)
        y.append(val_y)
        count += 1
        attempts += 1
        
    return np.array(x), np.array(y)

def calculate_coefficient(x, y):
    """计算 Pearson 相关系数"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    matrix = np.corrcoef(x, y)
    return matrix[0, 1]

def plot_correlation_grid(plain_path, cipher_path, save_path, csv_data_list):
    """绘制 2x4 布局的相关性分析图，并收集数据。"""
    # 1. 读取图像
    img_plain = cv2.imread(plain_path)
    img_cipher = cv2.imread(cipher_path)

    # 双重检查：如果读取失败则跳过
    if img_plain is None or img_cipher is None:
        return

    # 2. 预处理
    plain_rgb = cv2.cvtColor(img_plain, cv2.COLOR_BGR2RGB)
    cipher_rgb = cv2.cvtColor(img_cipher, cv2.COLOR_BGR2RGB)
    plain_gray = cv2.cvtColor(img_plain, cv2.COLOR_BGR2GRAY)
    cipher_gray = cv2.cvtColor(img_cipher, cv2.COLOR_BGR2GRAY)

    # 3. 准备绘图
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=100)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.15)
    
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    
    # 第一列：原图展示
    axes[0, 0].imshow(plain_rgb)
    axes[0, 0].set_title("Plain Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(cipher_rgb)
    axes[1, 0].set_title("Cipher Image")
    axes[1, 0].axis('off')

    image_name = os.path.basename(plain_path)
    
    # 临时字典
    img_coeffs = {d: {'plain': 0.0, 'cipher': 0.0} for d in directions}

    # 后三列：循环处理三个方向
    for idx, direction in enumerate(directions):
        col = idx + 1
        
        # --- 明文 ---
        px, py = get_adjacent_pixels(plain_gray, direction, N_SAMPLES)
        p_coeff = calculate_coefficient(px, py)
        img_coeffs[direction]['plain'] = p_coeff
        
        # 参数 s (size) 控制点的大小
        # 参数 c (color) 控制颜色
        # 参数 alpha 控制透明度
        ax_p = axes[0, col]
        ax_p.scatter(px, py, s=1, c='#0033cc', alpha=0.9)
        ax_p.set_xlim([0, 255])
        ax_p.set_ylim([0, 255])
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_title(f"{direction}\nCC: {p_coeff:.4f}") # 在标题显示相关系数

        # --- 密文 ---
        cx, cy = get_adjacent_pixels(cipher_gray, direction, N_SAMPLES)
        c_coeff = calculate_coefficient(cx, cy)
        img_coeffs[direction]['cipher'] = c_coeff

        ax_c = axes[1, col]
        ax_c.scatter(cx, cy, s=1, c='#cc0000', alpha=0.9)
        ax_c.set_xlim([0, 255])
        ax_c.set_ylim([0, 255])
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        ax_c.set_title(f"CC: {c_coeff:.4f}")

    # --- 将数据添加到列表 ---
    for direction in directions:
        csv_data_list.append({
            'image_name': image_name,   # 键名保持 'image_name'
            'direction': direction,
            'plain': f"{img_coeffs[direction]['plain']:.4f}",
            'cipher': f"{img_coeffs[direction]['cipher']:.4f}"
        })

    # 保存图片
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Processed: {image_name}")

def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    # 获取文件列表
    plain_files = os.listdir(PLAIN_DIR)
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    plain_files = [f for f in plain_files if os.path.splitext(f)[1].lower() in valid_exts]
    plain_files.sort()

    # ---------------------------------------------------------
    # 修改点 1：只筛选那些“既在明文文件夹，又在密文文件夹”的图片
    # 这样就自动跳过了那些你还没处理的图片，不用报错，也不用打印 Warning
    # ---------------------------------------------------------
    cipher_files_set = set(os.listdir(CIPHER_DIR)) # 获取密文目录所有文件放入集合，查询速度快
    
    target_files = []
    for f in plain_files:
        if f in cipher_files_set:
            target_files.append(f)
            
    print(f"Total plain images: {len(plain_files)}")
    print(f"Found cipher images: {len(target_files)} (Skipping {len(plain_files) - len(target_files)} unprocessed images)")

    csv_rows = []

    # 循环处理筛选后的文件
    for filename in target_files:
        p_path = os.path.join(PLAIN_DIR, filename)
        c_path = os.path.join(CIPHER_DIR, filename)
        
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(RESULT_DIR, f"{name}_correlation.png")
        
        plot_correlation_grid(p_path, c_path, save_path, csv_rows)

    # 4. 写入 CSV 文件
    # ---------------------------------------------------------
    # 修改点 2：Headers 必须和字典里的 Key 完全一致
    # 之前是 'image name' (空格)，现在统一改为 'image_name' (下划线)
    # ---------------------------------------------------------
    headers = ['image_name', 'direction', 'plain', 'cipher']
    
    direction_order = {'Horizontal': 1, 'Vertical': 2, 'Diagonal': 3}
    csv_rows.sort(key=lambda x: (x['image_name'], direction_order.get(x['direction'], 0)))

    if len(csv_rows) > 0:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nAnalysis complete.")
        print(f"CSV saved to:   {CSV_PATH}")
    else:
        print("\nNo data processed. Please check if filenames match in plain and cipher folders.")

if __name__ == "__main__":
    main()