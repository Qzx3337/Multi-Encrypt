import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math

class VerificationConfig:
    """配置类：用于存储阈值和权重"""
    # MS-SSIM 判定为 True 的阈值 (0-1之间，1为完全一致)
    MS_SSIM_THRESHOLD = 0.98 
    # MS-SSIM 计算时的权重 (5个尺度，参考 Wang et al. 的论文权重，或者使用平均值)
    # 这里使用简化的等权重或经验权重来模拟多尺度感知
    MS_SSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

def calculate_zncc(img1, img2):
    """
    计算归一化互相关 (Zero-Normalized Cross-Correlation)
    返回范围: -1 到 1，1 表示最强正相关
    """
    # 展平图像数组并转为浮点型以防止溢出
    img1_flat = img1.flatten().astype('float32')
    img2_flat = img2.flatten().astype('float32')
    
    # 减去均值
    img1_flat -= np.mean(img1_flat)
    img2_flat -= np.mean(img2_flat)
    
    # 计算分子 (协方差)
    numerator = np.sum(img1_flat * img2_flat)
    
    # 计算分母 (标准差的乘积)
    denominator = np.sqrt(np.sum(img1_flat**2)) * np.sqrt(np.sum(img2_flat**2))
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def calculate_ms_ssim(img1, img2):
    """
    计算多尺度结构相似性 (MS-SSIM)
    通过对图像金字塔进行下采样，在不同分辨率下计算 SSIM 并加权组合。
    """
    if img1.shape != img2.shape:
        return 0.0

    weights = VerificationConfig.MS_SSIM_WEIGHTS
    mssim_val = 1.0
    mcs_array = [] # 存储每一层的相似度
    
    current_img1 = img1
    current_img2 = img2
    
    # 确保图像足够大以进行多次下采样，否则减少层数
    levels = len(weights)
    min_dim = min(img1.shape[0], img1.shape[1])
    # 如果图像太小，无法进行预设层数的缩放，则动态调整层数
    possible_levels = int(math.log(min_dim / 7, 2)) # 7是SSIM窗口大小的典型值
    real_levels = min(levels, possible_levels)
    
    if real_levels < 1:
        # 图像极小，直接计算单层SSIM
        return ssim(img1, img2, data_range=255)

    for i in range(real_levels):
        # 计算当前尺度的 SSIM
        # skimage 的 ssim 默认使用了高斯加权窗口
        score = ssim(current_img1, current_img2, data_range=255)
        
        mcs_array.append(score)
        
        # 下采样图像 (Pyramid Down)
        current_img1 = cv2.pyrDown(current_img1)
        current_img2 = cv2.pyrDown(current_img2)

    # 简化的 MS-SSIM 计算：对各层 SSIM 进行加权乘积或加权平均
    # 标准 MS-SSIM 是乘积形式，但在实际工程测试中，加权平均也很常用。
    # 这里采用加权平均以获得更直观的线性分数，更便于调试。
    final_msssim = np.average(mcs_array, weights=weights[:real_levels])
    
    return final_msssim

def verify_single_project(source_dir, result_dir, project_name):
    """
    对比单一被加密项目的子模块。
    
    参数:
        source_dir (str): 原始文件目录
        result_dir (str): 解密后结果文件目录
        project_name (str): 项目名称 (不带后缀，例如 "data_01")
        
    返回:
        bool: 
            - TXT 必须完全一致
            - PNG 的 MS-SSIM 必须大于阈值
            - 如果文件缺失或尺寸不匹配，返回 False
    """
    print(f"--- 开始测试项目: {project_name} ---")
    
    # 构建文件路径
    src_txt_path = os.path.join(source_dir, f"{project_name}.txt")
    res_txt_path = os.path.join(result_dir, f"{project_name}.txt")
    src_png_path = os.path.join(source_dir, f"{project_name}.png")
    res_png_path = os.path.join(result_dir, f"{project_name}.png")

    # 1. 检查文件是否存在
    for p in [src_txt_path, res_txt_path, src_png_path, res_png_path]:
        if not os.path.exists(p):
            print(f"[失败] 文件缺失: {p}")
            return False

    # 2. 对比 TXT 文件 (严格一致)
    try:
        with open(src_txt_path, 'rb') as f1, open(res_txt_path, 'rb') as f2:
            txt_src_content = f1.read()
            txt_res_content = f2.read()
            
        if txt_src_content == txt_res_content:
            print(f"[TXT] 内容一致: Pass")
            txt_is_valid = True
        else:
            print(f"[TXT] 内容不一致: Fail")
            txt_is_valid = False
    except Exception as e:
        print(f"[TXT] 读取错误: {e}")
        return False

    # 3. 对比 PNG 文件 (指标评价)
    # 读取为灰度图进行结构对比 (IMREAD_GRAYSCALE)，通常评价结构相似性不需要色彩
    # 如果你的加密涉及色彩保真度，可以去掉 flag 使用默认读取，但在计算指标时需转灰度或分通道计算
    img_src = cv2.imread(src_png_path, cv2.IMREAD_GRAYSCALE)
    img_res = cv2.imread(res_png_path, cv2.IMREAD_GRAYSCALE)

    if img_src is None or img_res is None:
        print("[PNG] 图像读取失败 (可能文件损坏)")
        return False

    if img_src.shape != img_res.shape:
        print(f"[PNG] 尺寸不匹配: 原图{img_src.shape} vs 结果{img_res.shape}")
        return False

    # 计算指标
    val_psnr = psnr(img_src, img_res, data_range=255)
    val_zncc = calculate_zncc(img_src, img_res)
    val_msssim = calculate_ms_ssim(img_src, img_res)

    # 控制台输出指标
    print(f"[PNG 指标详情] {project_name}.png")
    print(f"  > PSNR    : {val_psnr:.4f} dB (通常 > 30dB)")
    print(f"  > ZNCC    : {val_zncc:.6f} (通常 > 0.99)")
    print(f"  > MS-SSIM : {val_msssim:.6f} (阈值: {VerificationConfig.MS_SSIM_THRESHOLD})")

    # 判定逻辑：TXT 必须 True，MS-SSIM 必须达标
    png_is_valid = val_msssim >= VerificationConfig.MS_SSIM_THRESHOLD
    
    if not png_is_valid:
        print(f"[PNG] 视觉一致性未达标 (MS-SSIM < {VerificationConfig.MS_SSIM_THRESHOLD})")
    
    final_result = txt_is_valid and png_is_valid
    
    if final_result:
        print(f"--- 项目 {project_name} 测试通过 ---\n")
    else:
        print(f"--- 项目 {project_name} 测试失败 ---\n")

    return final_result

def verify_directory(source_dir, result_dir):
    """
    主测试函数：遍历目录并对比所有项目。
    
    参数:
        source_dir (str): 原始文件目录路径
        result_dir (str): 解密结果目录路径
        
    返回:
        bool: 如果所有项目都通过测试，返回 True，否则返回 False
    """
    print("========================================")
    print("      开始批量验证加密/解密一致性")
    print("========================================\n")

    if not os.path.exists(source_dir) or not os.path.exists(result_dir):
        print("错误：输入目录不存在。")
        return False

    # 1. 扫描原始目录，获取所有项目名称
    # 假设项目由 name.txt 和 name.png 组成，我们以 .txt 为基准提取项目名
    files = os.listdir(source_dir)
    project_names = set()
    
    for f in files:
        if f.endswith('.txt'):
            base_name = os.path.splitext(f)[0]
            # 确认该项目是否有对应的 png
            if os.path.exists(os.path.join(source_dir, base_name + ".png")):
                project_names.add(base_name)

    if not project_names:
        print("警告：在原始目录中未发现完整的项目对 (txt+png)。")
        return False

    sorted_projects = sorted(list(project_names))
    print(f"发现 {len(sorted_projects)} 个待测试项目。\n")

    total_pass = 0
    total_fail = 0
    failed_projects = []

    # 2. 遍历测试
    for proj in sorted_projects:
        is_success = verify_single_project(source_dir, result_dir, proj)
        if is_success:
            total_pass += 1
        else:
            total_fail += 1
            failed_projects.append(proj)

    # 3. 最终汇总输出
    print("================ 验证汇总 ================")
    print(f"总项目数 : {len(sorted_projects)}")
    print(f"通过     : {total_pass}")
    print(f"失败     : {total_fail}")
    
    if total_fail > 0:
        print(f"失败项目列表: {failed_projects}")
        print("最终结果 : [不通过] False")
        return False
    else:
        print("最终结果 : [完全一致] True")
        return True

# --- 使用示例 ---
if __name__ == "__main__":
    # 你可以在这里修改路径进行本地直接测试
    # 假设你的目录结构如下
    src_path = r"./source_files"
    res_path = r"./extracted_files"
    
    # 确保目录存在再运行，防止报错
    if os.path.exists(src_path) and os.path.exists(res_path):
        verify_directory(src_path, res_path)
    else:
        print("请在代码底部 'if __name__ == ...' 中配置正确的测试路径以运行示例。")

