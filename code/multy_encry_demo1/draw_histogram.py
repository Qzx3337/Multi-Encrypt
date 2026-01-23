import os
import cv2
import matplotlib.pyplot as plt


def draw_histogram_for_file(image_path, save_path):
    """
    读取单个图像文件，计算其RGB直方图并保存统计图。
    修改点：纵坐标使用对数刻度，且填充曲线下面积。
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to open image {image_path}")
        return

    plt.figure()
    # plt.title("RGB Histogram (Log Scale)")
    plt.title("RGB Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    # 【修改点 1】：设置纵坐标为对数坐标
    # 这句话会让 Y 轴的刻度变成 10^0, 10^1, 10^2... 
    # 对于加密后的均匀分布或者差异巨大的像素分布，对数坐标能看得更清楚。
    # plt.yscale('log') 

    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        # 计算直方图
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        
        # 将 opencv 计算出的 (256, 1) 数组展平为一维，方便绘图
        hist_data = hist.flatten()
        
        # 绘制曲线 (保留轮廓线，看起来更清晰)
        # plt.plot(hist_data, color=color, linewidth=0.2)
        
        # 【修改点 2】：填充线下颜色
        # range(256): 指定 x 轴范围是 0 到 255
        # hist_data: 指定 y 轴数据
        # alpha=0.5: 设置透明度为 0.5，这样红绿蓝重叠的部分也能看清，而不是互相覆盖
        plt.fill_between(range(256), hist_data, color=color, alpha=0.9)

        plt.xlim([0, 256])

    plt.savefig(save_path)
    plt.close()


def process_histogram_folder(source_dir, output_dir):
    """
    遍历 source_dir 下的所有图片，绘制直方图并保存到 output_dir。
    """
    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. 遍历源文件夹
    files = os.listdir(source_dir)
    count = 0
    
    print(f"Processing histograms for folder: {source_dir} ...")
    
    for filename in files:
        # 简单过滤图片文件，可根据需要增加 png 等格式
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(source_dir, filename)
            
            # 为了区分，输出的文件名可以加个 _hist 后缀，或者保持原名（这里保持原名，但扩展名可能需要注意）
            # 这里直接使用原文件名，保存为 png 格式（matplotlib 默认支持）
            output_filename = f"{os.path.splitext(filename)[0]}_hist.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                draw_histogram_for_file(input_path, output_path)
                count += 1
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"Finished. Generated {count} histograms in '{output_dir}'.\n")

if __name__ == "__main__":
    # 定义文件夹路径
    
    # 1. 原始图片及其直方图存放目录
    plain_img_dir = "data1"
    plain_hist_dir = "plain_hist"
    
    # 2. 加密图片及其直方图存放目录
    cipher_img_dir = "cipher_img"
    cipher_hist_dir = "cipher_hist"

    # 执行批量绘图：处理未加密图片
    if os.path.exists(plain_img_dir):
        process_histogram_folder(plain_img_dir, plain_hist_dir)
    else:
        print(f"Warning: Directory '{plain_img_dir}' does not exist.")

    # 执行批量绘图：处理加密后图片
    if os.path.exists(cipher_img_dir):
        process_histogram_folder(cipher_img_dir, cipher_hist_dir)
    else:
        print(f"Warning: Directory '{cipher_img_dir}' does not exist.")

