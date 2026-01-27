import cv2

import matplotlib.pyplot as plt

# ===================== 测试加密效果 ======================== #

def calculate_correlation(image, direction):
    if direction == 'horizontal':
        return np.corrcoef(image[:, :-1].ravel(), image[:, 1:].ravel())[0, 1]
    elif direction == 'vertical':
        return np.corrcoef(image[:-1, :].ravel(), image[1:, :].ravel())[0, 1]
    elif direction == 'diagonal':
        h, w = image.shape[:2]
        diag = np.array([image[i, i] for i in range(min(h, w))])
        return np.corrcoef(diag[:-1], diag[1:])[0, 1]


def plot_correlation_distribution(image, channel, direction):
    if channel == 'B':
        channel_image = image[:, :, 0]
    elif channel == 'G':
        channel_image = image[:, :, 1]
    elif channel == 'R':
        channel_image = image[:, :, 2]

    if direction == 'horizontal':
        x = channel_image[:, :-1].ravel()
        y = channel_image[:, 1:].ravel()
    elif direction == 'vertical':
        x = channel_image[:-1, :].ravel()
        y = channel_image[1:, :].ravel()
    elif direction == 'diagonal':
        h, w = channel_image.shape
        diag = np.array([channel_image[i, i] for i in range(min(h, w))])
        x = diag[:-1]
        y = diag[1:]

    plt.scatter(x, y, s=1, alpha=0.5)
    plt.title(f'{direction.capitalize()} Direction Correlation Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Adjacent Pixel Value')
    plt.show()


def generate_dis_pic(plain_path, cipher_path):
    """
    生成原图和加密图的相关系数及相关分布图
    """
    # 加密前图像路径
    original_image_path = plain_path
    # 加密后图像路径
    encrypted_image_path = cipher_path

    # 读取图像
    original_image = cv2.imread(original_image_path)
    encrypted_image = cv2.imread(encrypted_image_path)

    # 将BGR图像转换为RGB图像（因为OpenCV默认读取的是BGR格式）
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    encrypted_image_rgb = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

    # 计算相关系数
    correlations_original = {
        'horizontal': calculate_correlation(original_image_rgb, 'horizontal'),
        'vertical': calculate_correlation(original_image_rgb, 'vertical'),
        'diagonal': calculate_correlation(original_image_rgb, 'diagonal')
    }

    correlations_encrypted = {
        'horizontal': calculate_correlation(encrypted_image_rgb, 'horizontal'),
        'vertical': calculate_correlation(encrypted_image_rgb, 'vertical'),
        'diagonal': calculate_correlation(encrypted_image_rgb, 'diagonal')
    }

    # 打印相关系数
    print("Original Image Correlations:")
    for direction, correlation in correlations_original.items():
        print(f"{direction.capitalize()}: {correlation}")

    print("\nEncrypted Image Correlations:")
    for direction, correlation in correlations_encrypted.items():
        print(f"{direction.capitalize()}: {correlation}")

    # 绘制相关分布图
    plot_correlation_distribution(original_image_rgb, 'R', 'horizontal')
    plot_correlation_distribution(original_image_rgb, 'R', 'vertical')
    plot_correlation_distribution(original_image_rgb, 'R', 'diagonal')

    plot_correlation_distribution(encrypted_image_rgb, 'R', 'horizontal')
    plot_correlation_distribution(encrypted_image_rgb, 'R', 'vertical')
    plot_correlation_distribution(encrypted_image_rgb, 'R', 'diagonal')



def load_image(path, channel='B'):
    img = Image.open(path)
    if channel == 'B':
        # 转换为灰度图像
        img = img.convert('L')
    else:
        # 提取特定颜色通道（未实现）
        r, g, b = img.split()
        if channel == 'R':
            img = r
        elif channel == 'G':
            img = g
        elif channel == 'B':
            img = b
    return np.array(img)


def calculate_correlation_distribution(img):
    h, w = img.shape

    horizontal_pairs = [(img[i, j], img[i, (j + 1) % w]) for i in range(h) for j in range(w - 1)]
    vertical_pairs = [(img[i, j], img[(i + 1) % h, j]) for i in range(h - 1) for j in range(w)]
    diagonal_pairs = [(img[i, j], img[i + 1, j + 1]) for i in range(h - 1) for j in range(w - 1) if
                      i < h - 1 and j < w - 1]

    return horizontal_pairs, vertical_pairs, diagonal_pairs


def calculate_correlation_coefficients(pairs_list):
    coefficients = []
    for pairs in pairs_list:
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]

        # 计算相关性系数
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_coefficient = correlation_matrix[0, 1]
        coefficients.append(correlation_coefficient)

    return coefficients


def print_results(coefficients):
    print("Image\t\tHorizontal\tVertical\tDiagonal")
    print(f"Original Image\t{coefficients[0]:.5f}\t\t{coefficients[1]:.5f}\t\t{coefficients[2]:.5f}")


def plot_correlation_distributions(pairs_list):
    fig, axs = plt.subplots(1, len(pairs_list), figsize=(15, 5))

    for ax, pairs in zip(axs, pairs_list):
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]

        ax.scatter(x_values, y_values, s=1, c='blue', alpha=0.5)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)

    plt.tight_layout()
    plt.show()


def process_and_visualize(image_path):
    img = load_image(image_path, channel='B')
    horizontal_pairs, vertical_pairs, diagonal_pairs = calculate_correlation_distribution(img)
    pairs_list = [horizontal_pairs, vertical_pairs, diagonal_pairs]

    # 计算相关性系数
    coefficients = calculate_correlation_coefficients(pairs_list)

    # 输出结果
    print_results(coefficients)

    # 绘制图表
    plot_correlation_distributions(pairs_list)


import numpy as np
from PIL import Image


def calculate_entropy(image_path):
    # 打开图像并转换为灰度图像
    img = Image.open(image_path).convert('L')
    # 转换为numpy数组
    img_array = np.array(img)

    # 计算直方图
    histogram, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # 将直方图转换为概率分布
    probabilities = histogram / float(np.sum(histogram))

    # 过滤掉零概率，避免log(0)错误
    probabilities = probabilities[probabilities > 0]

    # 计算信息熵
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def NPCR(img1, img2):
    # opencv颜色通道顺序为BGR
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape

    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((R1 != R2), return_counts=True)
    R_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)
    ar, num = np.unique((G1 != G2), return_counts=True)
    G_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)
    ar, num = np.unique((B1 != B2), return_counts=True)
    B_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)

    return R_npcr, G_npcr, B_npcr


def UACI(img1_path: str, img2_path: str):
    """    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    w, h, _ = img1.shape
    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 元素为uint8类型取值范围：0到255
    # print(R1.dtype)

    # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)

    sumR = np.sum(abs(R1 - R2))
    sumG = np.sum(abs(G1 - G2))
    sumB = np.sum(abs(B1 - B2))
    R_uaci = sumR / 255 / (w * h)
    G_uaci = sumG / 255 / (w * h)
    B_uaci = sumB / 255 / (w * h)

    return R_uaci, G_uaci, B_uaci

