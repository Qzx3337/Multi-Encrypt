from PIL import Image
import numpy as np
import cv2
import os
import math


# 配置
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'source_files')
OUT_DIR = os.path.join(BASE_DIR, 'embedded_files')
MAX_CHARS_DCT = 64  # 频域嵌入建议阈值
END_MARKER = '<END>'
MARKER_REPEATS = 3
ALPHA = 30

def embed_text_dct(src_img_path, txt_path, embedded_img_path):
    """
    使用DCT频域盲水印方法，将文本嵌入图像中，结果图像保存到指定路径。

    Args:
        src_img_path (str): 原始图像路径（PNG格式）
        txt_path (str): 待嵌入文本文件路径
        embedded_img_path (str): 嵌入后图像保存路径（PNG格式）

    Returns:
        bool: 嵌入成功返回True

    备注:
        - 仅在亮度通道嵌入文本，保持色彩信息不变
        - 使用简单的量化索引奇偶性调整中频DCT系数实现盲水印
    """
    # 读取文本并截断
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    if len(text) > MAX_CHARS_DCT:
        text = text[:MAX_CHARS_DCT]
    payload = text + (END_MARKER * MARKER_REPEATS)
    data_bytes = payload.encode('utf-8')
    data_bits = []
    for byte in data_bytes:
        for i in range(8)[::-1]:
            data_bits.append((byte >> i) & 1)

    # 读取图像并保持彩色：在YCbCr色彩空间的Y通道（亮度）中嵌入，随后还原为RGB
    img = Image.open(src_img_path).convert('YCbCr')
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    # 分离通道
    y_chan = arr[:, :, 0]
    cb_chan = arr[:, :, 1]
    cr_chan = arr[:, :, 2]
    block_size = 8
    bit_idx = 0
    for yy in range(0, h - block_size + 1, block_size):
        for xx in range(0, w - block_size + 1, block_size):
            block = y_chan[yy:yy+block_size, xx:xx+block_size]
            dct_block = cv2.dct(block)
            
            # 选定中频系数位置（如[3,3]）
            if bit_idx < len(data_bits):
                coeff = dct_block[3,3]
                
                bit = data_bits[bit_idx]
                
                # 1. 计算当前系数对应的是第几个区间
                q = round(coeff / ALPHA)
                
                # 2. 检查该区间的奇偶性是否与 bit 一致
                if q % 2 != bit:
                    # 如果不一致，需要移动到相邻的正确区间
                    # 比较移到 q+1 还是 q-1 离原值更近，以减少对图像破坏
                    if abs((q + 1) * ALPHA - coeff) < abs((q - 1) * ALPHA - coeff):
                        q = q + 1
                    else:
                        q = q - 1
                
                # 3. 将系数强制设定为区间的中心值
                dct_block[3,3] = q * ALPHA

                bit_idx += 1
            
            y_chan[yy:yy+block_size, xx:xx+block_size] = cv2.idct(dct_block)

            if bit_idx >= len(data_bits):
                break
        if bit_idx >= len(data_bits):
            break
    # 合并通道并转换回RGB保存
    y_chan = np.clip(y_chan, 0, 255)
    out_arr = np.stack([y_chan, cb_chan, cr_chan], axis=2).astype(np.uint8)
    out_img = Image.fromarray(out_arr, 'YCbCr').convert('RGB')
    out_img.save(embedded_img_path)
    return True



# 新主函数：批量处理所有txt文件，使用DCT频域盲水印嵌入
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in os.listdir(SRC_DIR):
        if not fname.lower().endswith('.txt'):
            continue
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(SRC_DIR, fname)
        # 选择同名png为载体，否则报错
        src_img_path = os.path.join(SRC_DIR, base + '.png')
        if not os.path.exists(src_img_path):
            print(f'Skip {fname}: no matching image {base}.png')
            continue
        out_path = os.path.join(OUT_DIR, base + '.png')
        try:
            embed_text_dct(src_img_path, txt_path, out_path)
            print(f'Embedded "{fname}" -> {out_path}')
        except Exception as e:
            print(f'Failed to embed {fname}: {e}')


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print('Pillow is required: pip install pillow')
