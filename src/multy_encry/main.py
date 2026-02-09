
import os
import cv2
import pandas as pd
import glob

from MultimodalPacker import MultimodalPacker
from MultiEncryptor import MultiEncryptor

# 更改数据集
# DATABASE_NAME = "zky"
# DATABASE_NAME = "hyper_kvasir"
DATABASE_NAME = "test"
# 更改实验
EXP_NAME = "w507" 

# data\zky
PLAIN_DIR = os.path.join("data", DATABASE_NAME)
# experiments\w502\zky\cipher
CIPHER_DIR = os.path.join("experiments", EXP_NAME, DATABASE_NAME, "cipher")
OUTPUT_DIR = os.path.join("experiments", EXP_NAME, DATABASE_NAME, "decry")

MODEL_PATH = os.path.join("experiments", "exp_lorenz", "lorenz_f2_lr5en5_s1m.zip")

def run_integrated_test(plain_dir, cipher_dir, output_dir, k=5):
    """
    批量可行性测试函数
    """
    print(f"--- Starting Integrated Test (Top {k} items) ---")
    
    # 1. 准备环境
    csv_path = os.path.join(plain_dir, "messages.csv")
    img_dir = os.path.join(plain_dir, "images")
    if not (os.path.exists(csv_path) and os.path.exists(img_dir)):
        raise RuntimeError("[error] 明文文件夹不存在")
    os.makedirs(cipher_dir, exist_ok=True)
    images_output_subdir = os.path.join(output_dir, "images")
    os.makedirs(images_output_subdir, exist_ok=True)
    results_list = []
    packer = MultimodalPacker()
    encryptor = MultiEncryptor(model_path=MODEL_PATH)
    
    # 2. 读取数据索引
    # 假设CSV没有表头或第一列有效，这里按行读取
    # 也可以用 pandas: df = pd.read_csv(csv_path)
    try:
        df = pd.read_csv(csv_path, header=0) # header=0表示第一行为表头，根据实际情况调整
        # iloc[:, 3:] 提取所有行和第四列开始到最后一列
        # fillna('') 将表格的空位置代换为"" 不然会出现 nan
        # apply(lambda x: ', '.join(x), axis=1) 连缀所有列的时候, 使用', ' 分割
        texts = df.iloc[:, 3:].fillna('').astype(str).apply(lambda x: ', '.join(x), axis=1).tolist()
    except Exception as e:
        print(f"[ERROR] Error reading CSV: {e}")
        return

    # 获取图片列表并排序，确保与CSV行对应
    img_files = glob.glob(os.path.join(img_dir, '*.tiff'))
    img_files.sort()

    # 校验数量
    num_items = min(len(texts), len(img_files), k)
    print(f" Found {len(texts)} texts and {len(img_files)} images. Processing first {num_items} items.")

    # 3. 循环处理
    for i in range(num_items):
        
        # 每次处理开始时空一行方便查看信息
        print()
        
        # A. 读取原始数据
        original_text = texts[i]
        img_path = img_files[i]
        base_name = os.path.basename(img_path) # 获取文件名，如 "001.tiff"
        original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # UNCHANGED 以保留原图所有通道(如Alpha)
        
        
        if original_img is None:
            print(f"  Error: Could not read image {img_path}")
            continue
            
        # print(f"  Original Image Shape: {original_img.shape}")
        # print(f"  Original Text Len: {len(original_text)}")

        # B. 打包 (Pack)
        # 此时 packed_matrix 是在内存中的 (H, W, C+1) 矩阵
        packed_matrix = packer.pack(original_img, original_text)
        # print(f"  Packed Matrix Shape: {packed_matrix.shape} (Channel count increased by 1)")
        
        # --- 模拟：这里是 main.py 中传递给 Encryptor 的地方 ---
        cipher_save_name = os.path.join(cipher_dir, base_name)
        decrypted_matrix = encryptor.encry_and_decry_one_mat(packed_matrix, cipher_save_name)
        # 假设解密完美还原，直接将 packed_matrix 传给解包逻辑
        # decrypted_matrix_simulation = packed_matrix 
        # ---------------------------------------------------

        # C. 解包 (Unpack)
        restored_img, restored_text = packer.unpack(decrypted_matrix)
        
        # D. 验证与保存
        # 验证文本
        if original_text == restored_text:
            print(f"  [SUCCESS] Text of item {i} recovered perfectly.")
        else:
            # 如果发生了截断，这里会不等
            if len(restored_text) < len(original_text):
                 print(f"  [INFO] Text was truncated (expected behavior if too long).")
            else:
                 print(f"  [FAIL] Text mismatch!")
        
        # 验证图像形状
        if original_img.shape == restored_img.shape:
             print("  [SUCCESS] Image shape matches.")
        else:
             print("  [FAIL] Image shape mismatch.")

        # 1. 保存图片到 images 子目录
        save_img_path = os.path.join(images_output_subdir, base_name)
        cv2.imwrite(save_img_path, restored_img)
        print(f"  Saved decrypted image to: {save_img_path}")
        
        # 2. 收集文本信息到内存列表 (不直接写文件)
        results_list.append({
            "filename": base_name,      # 记录对应的图片文件名作为索引
            "restored_text": restored_text
        })

    # 4. 循环结束后：统一保存 CSV
    if results_list:
        csv_output_path = os.path.join(output_dir, "message.csv")
        
        # 使用 Pandas 创建 DataFrame 并保存
        result_df = pd.DataFrame(results_list)
        
        # encoding='utf-8-sig': 关键！
        # 加上 sig 签名可以确保 Excel 在 Windows 上打开中文不乱码
        result_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[SUCCESS] All text data saved to: {csv_output_path}")
    
            
    print("\n--- Integrated Test Completed ---")


if __name__=="__main__":
    run_integrated_test(PLAIN_DIR, 
                        CIPHER_DIR, 
                        OUTPUT_DIR, 
                        3)
