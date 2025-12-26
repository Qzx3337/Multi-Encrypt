# 03_evaluate.py
import os

# === 配置路径 ===
ORIGIN_DIR = 'source_files'     # 原始目录 (标准答案)
DECODED_DIR = 'decoded_results' # 解码目录 (待阅卷)

def evaluate_quality():
    print("--- 开始效果评估 ---")
    
    # 获取原始文件列表
    origin_files = [f for f in os.listdir(ORIGIN_DIR) if f.endswith('.txt')]
    
    total_count = len(origin_files)
    match_count = 0
    
    if total_count == 0:
        print("未发现原始文本文件，无法评估。")
        return

    print(f"{'文件名':<20} | {'状态':<10} | {'详细信息'}")
    print("-" * 60)

    for txt_file in origin_files:
        origin_path = os.path.join(ORIGIN_DIR, txt_file)
        decoded_path = os.path.join(DECODED_DIR, txt_file)
        
        # 1. 读取原始文字
        with open(origin_path, 'r', encoding='utf-8') as f:
            origin_text = f.read().strip()
            
        # 2. 读取解码文字
        if not os.path.exists(decoded_path):
            print(f"{txt_file:<20} | 缺失 | 解码目录中未找到该文件")
            continue
            
        with open(decoded_path, 'r', encoding='utf-8') as f:
            decoded_text = f.read().strip() # 盲水印提取可能会有空字符，strip很重要
            
        # 3. 对比
        # 盲水印提取的文字长度可能因为空字节填充而不一致，
        # 但核心内容应该包含在内。这里做精确匹配或包含匹配。
        if origin_text == decoded_text:
            status = "完美匹配"
            match_count += 1
        elif origin_text in decoded_text:
            status = "包含匹配" # 提取出的文字末尾可能带有乱码或填充符
            match_count += 1
        else:
            status = "不匹配"
        
        info = f"原长:{len(origin_text)} / 解长:{len(decoded_text)}"
        print(f"{txt_file:<20} | {status:<10} | {info}")

    # 总结
    print("-" * 60)
    print(f"评估完成。总数: {total_count}, 成功(完美/包含): {match_count}")
    print(f"成功率: {match_count/total_count*100:.2f}%")

if __name__ == '__main__':
    evaluate_quality()

    