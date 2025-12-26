from PIL import Image
import os

def convert_bmp_to_png(source_path, target_path=None):
    """
    将 BMP 图片转换为 PNG 格式
    """
    try:
        # 如果未指定目标路径，则在同一目录下更改后缀名
        if target_path is None:
            target_path = os.path.splitext(source_path)[0] + ".png"

        # 打开并保存
        with Image.open(source_path) as img:
            img.save(target_path, "PNG")
            
        print(f"成功: {source_path} -> {target_path}")
    except Exception as e:
        print(f"转换失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的文件名
    file_name = "source_files/LenaRGB.bmp"
    
    if os.path.exists(file_name):
        convert_bmp_to_png(file_name)
    else:
        print(f"未找到文件: {file_name}")