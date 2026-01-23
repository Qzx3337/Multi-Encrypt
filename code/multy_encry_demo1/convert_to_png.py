import os
import cv2

def batch_convert_jpg_to_png(root_dir, delete_source=False):
    """
    遍历指定目录，将所有JPG转换为PNG (无tqdm版本)。
    """
    # 1. 搜集所有需要转换的文件路径
    print(f"正在扫描目录: {root_dir} ...")
    jpg_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查后缀名 (不区分大小写)
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    
    total_files = len(jpg_files)
    print(f"扫描完成，共发现 {total_files} 个JPG文件，准备开始转换。\n")

    # 2. 开始转换循环
    for index, file_path in enumerate(jpg_files):
        try:
            # 打印进度: [当前序号/总数] 文件名
            print(f"[{index + 1}/{total_files}] 正在处理: {file_path}")

            # 读取图片
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"   >>> [警告] 无法读取文件，已跳过: {file_path}")
                continue
            
            # 构造新的文件名 (.jpg -> .png)
            base_name = os.path.splitext(file_path)[0]
            new_path = base_name + ".png"
            
            # 保存为PNG (压缩等级默认3)
            cv2.imwrite(new_path, img)
            
            # (可选) 删除源文件
            if delete_source:
                os.remove(file_path)
                print(f"   >>> 原文件已删除")
                
        except Exception as e:
            print(f"   >>> [错误] 处理出错: {e}")

    print("\n所有转换工作已完成。")

if __name__ == "__main__":
    # 请修改这里的路径
    data_path = "./CT_data" 
    
    # 建议先设为 False 运行一次，确认 PNG 生成无误后，再改为 True 删除 JPG
    batch_convert_jpg_to_png(data_path, delete_source=True)
