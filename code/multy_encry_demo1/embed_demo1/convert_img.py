from pathlib import Path
from PIL import Image

def convert_jpg_to_png_simple(data_dir='data'):
    """简化的转换函数"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"目录 '{data_dir}' 不存在！")
        return
    
    for jpg_file in data_path.rglob('*'):
        if jpg_file.suffix.lower() in ['.jpg', '.jpeg']:
            png_file = jpg_file.with_suffix('.png')
            
            if png_file.exists():
                print(f"跳过: {jpg_file.name} (PNG已存在)")
                continue
            
            try:
                with Image.open(jpg_file) as img:
                    # 转换为PNG
                    img.save(png_file, 'PNG')
                    print(f"转换: {jpg_file.name} -> {png_file.name}")
            except Exception as e:
                print(f"错误: 无法转换 {jpg_file.name} - {e}")

if __name__ == "__main__":
    convert_jpg_to_png_simple('source_files')