import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端
import matplotlib.pyplot as plt



def analyze_pixel_correlations(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read the image.")
        return

    height, width = image.shape

    # 初始化存储相关性的数组
    horizontal_correlations = []
    vertical_correlations = []
    diagonal_correlations = []

    # 计算水平方向的相关性
    for i in range(height - 1):
        for j in range(width - 1):
            horizontal_correlations.append((image[i, j], image[i, j + 1]))

    # 计算垂直方向的相关性
    for i in range(height - 1):
        for j in range(width - 1):
            vertical_correlations.append((image[i, j], image[i + 1, j]))

    # 计算对角方向的相关性
    for i in range(height - 1):
        for j in range(width - 1):
            diagonal_correlations.append((image[i, j], image[i + 1, j + 1]))

    # 绘制散点图
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(*zip(*horizontal_correlations), s=1, alpha=0.5)
    plt.title('Horizontal Direction')
    plt.xlabel('Pixel Value (i,j)')
    plt.ylabel('Pixel Value (i,j+1)')

    plt.subplot(1, 3, 2)
    plt.scatter(*zip(*vertical_correlations), s=1, alpha=0.5)
    plt.title('Vertical Direction')
    plt.xlabel('Pixel Value (i,j)')
    plt.ylabel('Pixel Value (i+1,j)')

    plt.subplot(1, 3, 3)
    plt.scatter(*zip(*diagonal_correlations), s=1, alpha=0.5)
    plt.title('Diagonal Direction')
    plt.xlabel('Pixel Value (i,j)')
    plt.ylabel('Pixel Value (i+1,j+1)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_pixel_correlations("D:/pythonmain/opencv/encryanddecry/test.jpg")
    plt.tight_layout()
    plt.savefig('output.png')  # 保存图像