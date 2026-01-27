import os
from chaos import encrypt, decrypt, generate_seed
import cv2


def check_decryption_pixel(plain_path, decrypted_path):
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    if plain_img.shape != decrypted_img.shape:
        return False
    difference = cv2.absdiff(plain_img, decrypted_img)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    else:
        return False


def check_decryption_psnr(plain_path: str, decrypted_path: str):
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    
    if plain_img.shape != decrypted_img.shape:
        return False

    # 计算 PSNR
    psnr_value = cv2.PSNR(plain_img, decrypted_img)
    
    # 设定阈值，比如 40dB，意味着差异极小
    if psnr_value > 40: 
        return True
    else:
        return False


def encrypt_and_decrypt_once(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None):
    if plain_path is None or cipher_path is None or decrypted_path is None:
        raise ValueError("File paths cannot be None.")
    seed = generate_seed()
    decry_key = encrypt(seed, plain_path, cipher_path)
    decrypt(decry_key, cipher_path, decrypted_path)
    if check_decryption_psnr(plain_path, decrypted_path):
        print("Decryption successful: The decrypted image matches the original.")
        if not check_decryption_pixel(plain_path, decrypted_path):
            print("Warning: Decryption verification failed: Pixel values do not match exactly.")
        else:
            print("Verification succeed: Pixel values match exactly.")
        return True
    else:
        print("Error: Decryption failed. The decrypted image does not match the original.")
        return False


def encrypt_and_decrypt(plain_path: str = None, cipher_path: str = None, decrypted_path: str = None):
    cnt = 0
    while True:
        cnt += 1
        print("********** Round: {} **********".format(cnt))
        flag = encrypt_and_decrypt_once(plain_path, cipher_path, decrypted_path)
        if flag:
            break
        elif cnt >= 10:
            raise Exception("Decryption failed after 10 attempts.")
            

def process_images_in_folder(source_dir, cipher_dir, decrypted_dir):
    """
    遍历 source_dir 下的所有 png 图片，进行加密和解密，
    并将结果分别保存到 cipher_dir 和 decrypted_dir。
    如果 decrypted_dir 中已存在同名文件，则跳过（增量更新）。
    """
    # 1. 确保输出目录存在，如果不存在则创建
    os.makedirs(cipher_dir, exist_ok=True)
    os.makedirs(decrypted_dir, exist_ok=True)

    # 2. 获取源目录下的所有文件
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    files = os.listdir(source_dir)
    count_processed = 0
    count_skipped = 0

    print(f"Starting batch processing in: {source_dir}\n" + "-"*40)

    for file_name in files:
        # 3. 过滤文件，只处理 png (忽略大小写)
        if file_name.lower().endswith('.png'):
            
            # 构造完整路径
            plain_path = os.path.join(source_dir, file_name)
            cipher_path = os.path.join(cipher_dir, file_name)
            decrypted_path = os.path.join(decrypted_dir, file_name)

            # 4. 核心逻辑：检查是否已经处理过
            # 如果解密文件夹里已经有了这个图，说明之前跑过，直接跳过
            print()
            if os.path.exists(decrypted_path):
                print(f"[SKIP] {file_name} already exists in decrypted folder.")
                count_skipped += 1
                continue

            # 5. 开始处理新图片
            print(f"[ACTION] Processing new image: {file_name}")
            try:
                # 调用你写好的带有重试机制的函数
                encrypt_and_decrypt(plain_path, cipher_path, decrypted_path)
                count_processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {file_name}. Reason: {str(e)}")
    
    print("-" * 40)
    print(f"Batch task finished. New Processed: {count_processed}, Skipped: {count_skipped}.")
    pass


if __name__ == "__main__":
    # 定义文件夹路径
    plain_folder = "pictures/plain_img"          # 未加密图片文件夹
    cipher_folder = "pictures/cipher_img"        # 加密后存放文件夹
    decrypted_folder = "pictures/decrypted_img"  # 解密后存放文件夹

    # 确保源文件夹存在，否则无法处理
    if os.path.exists(plain_folder):
        try:
            process_images_in_folder(plain_folder, cipher_folder, decrypted_folder)
        except Exception as e:
            print("An error occurred during batch processing:", str(e))
    else:
        print(f"Error: Source directory '{plain_folder}' not found.")

