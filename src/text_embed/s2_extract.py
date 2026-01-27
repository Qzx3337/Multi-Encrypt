import os
import numpy as np
import cv2
from PIL import Image

# 配置 (需保持与 embed.txt 一致)
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'embedded_files') # 这里默认读取嵌入后的文件目录
OUT_DIR = os.path.join(BASE_DIR, 'extracted_files')
END_MARKER = '<END>'
MARKER_REPEATS = 3
BLOCK_SIZE = 8
ALPHA = 30
# ALPHA = 20

def extract_text_dct(embedded_img_path, output_txt_path, output_img_path):
    """
    从 embedded_img_path 图像的 DCT 频域中提取文本，保存到 output_txt_path。
    同时将图像另存为 output_img_path 以满足接口要求。
    """
    try:
        # 1. 读取图像并转换为 YCbCr
        img = Image.open(embedded_img_path).convert('YCbCr')
        arr = np.array(img, dtype=np.float32)
        h, w, _ = arr.shape
        y_chan = arr[:, :, 0]

        extracted_bits = []
        
        # 2. 遍历 8x8 区块进行 DCT
        # 注意：这里的循环逻辑必须与嵌入时完全一致
        for yy in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
            for xx in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
                block = y_chan[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE]
                dct_block = cv2.dct(block)
                
                # 3. 提取 (3,3) 系数
                coeff = dct_block[3, 3]
                
                # 关键：使用 round() 抵抗 float->uint8->float 转换带来的量化噪音
                # 如果 coeff 是 10.9，round为11，%2为1。
                # val = int(round(coeff))
                val = int(round(coeff / ALPHA))
                extracted_bits.append(val % 2)

        # 4. 将比特流重组为字节
        # 嵌入顺序是 MSB first (byte >> i)
        extracted_bytes = bytearray()
        current_byte = 0
        bit_count = 0

        full_marker = END_MARKER * MARKER_REPEATS
        decoded_text = ""
        found_end = False

        for bit in extracted_bits:
            current_byte = (current_byte << 1) | bit
            bit_count += 1
            
            if bit_count == 8:
                extracted_bytes.append(current_byte)
                current_byte = 0
                bit_count = 0
                
                # 尝试解码并检查结束标记
                # 为了效率，每凑够一定字节检测一次，或者简单地每字节检测
                try:
                    # 尝试将当前所有字节解码为字符串
                    # ignore 错误是因为截断的 UTF-8 序列可能会报错
                    temp_text = extracted_bytes.decode('utf-8', errors='ignore')
                    if full_marker in temp_text:
                        decoded_text = temp_text.split(full_marker)[0]
                        found_end = True
                        break
                except:
                    pass
        
        if not found_end:
            # 如果没找到标记，尝试全部解码
            decoded_text = extracted_bytes.decode('utf-8', errors='ignore')
            print(f"Warning: End marker not found in {os.path.basename(embedded_img_path)}, text may be incomplete or garbage.")

        # 5. 保存结果
        # 保存文本
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        
        # 保存图像 (接口要求：输出一份png)
        # 这里直接保存读取的 RGB 版本，作为提取过程的证据/可视化
        img.convert('RGB').save(output_img_path)
        
        return True

    except Exception as e:
        print(f"Error extracting {embedded_img_path}: {e}")
        return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(SRC_DIR):
        print(f"Source directory '{SRC_DIR}' does not exist. Please run embed code first.")
        return

    print(f"Extracting from {SRC_DIR} to {OUT_DIR}...")

    for fname in os.listdir(SRC_DIR):
        if not fname.lower().endswith('.png'):
            continue
            
        base = os.path.splitext(fname)[0]
        src_img_path = os.path.join(SRC_DIR, fname)
        
        # 输出路径
        out_txt_path = os.path.join(OUT_DIR, base + '.txt')
        out_png_path = os.path.join(OUT_DIR, base + '.png') # 提取后的图像副本
        
        success = extract_text_dct(src_img_path, out_txt_path, out_png_path)
        
        if success:
            print(f'Extracted "{fname}" -> {out_txt_path}')

if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print('Requirements: pip install pillow numpy opencv-python')
        