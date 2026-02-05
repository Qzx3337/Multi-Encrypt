import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm # 推荐安装 tqdm 用于显示进度条: pip install tqdm

class TiffPreprocessor:
    """
    用于图像加密任务的预处理类。
    主要功能：批量将 PNG/JPG 转换为 TIFF 格式，保留原始通道数据。
    """

    def __init__(self, input_dir, output_dir):
        """
        初始化预处理器。
        :param input_dir: 输入图像文件夹路径
        :param output_dir: 输出 TIFF 文件夹路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 确保输出目录存在
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            print(f"[Info] 创建输出目录: {self.output_dir}")

    def convert_to_tiff(self):
        """
        读取输入目录下的所有 png/jpg/jpeg 文件，转换为 tiff 并保存到输出目录。
        注意：OpenCV 读取的默认通道顺序是 BGR。
        """
        # 支持的扩展名
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        # 获取所有图片文件
        files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in valid_extensions]
        
        if not files:
            print("[Warning] 输入目录中未找到支持的图像文件。")
            return

        print(f"[Start] 开始处理 {len(files)} 张图像...")
        
        success_count = 0
        
        # 使用 tqdm 显示进度条 (如果未安装 tqdm，可以直接用 for f in files:)
        for file_path in tqdm(files, desc="Converting"):
            try:
                # 1. 读取图像
                # cv2.IMREAD_UNCHANGED 是关键：
                # - 如果是 jpg (3通道)，读入 BGR
                # - 如果是 png (可能4通道)，读入 BGRA
                # - 避免默认的 cv2.imread 把 alpha 通道丢弃
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"[Error] 无法读取文件: {file_path.name}")
                    continue

                # 2. 检查通道数 (调试用)
                # print(f"Processing {file_path.name}: Shape {img.shape}")
                
                # 3. 构造输出路径，修改后缀为 .tiff
                output_filename = file_path.stem + ".tiff"
                output_path = self.output_dir / output_filename
                
                # 4. 保存为 TIFF
                # OpenCV 的 imwrite 支持多通道 TIFF 保存
                # 默认情况下，OpenCV 保存 TIFF 可能会使用 LZW 压缩 (无损)
                cv2.imwrite(str(output_path), img)
                
                success_count += 1
                
            except Exception as e:
                print(f"[Error] 处理文件 {file_path.name} 时出错: {e}")

        print(f"[Finished] 处理完成。成功转换: {success_count}/{len(files)}")
        print(f"[Note] 输出图像通道顺序默认为 OpenCV 的 BGR (或 BGRA)。")

    @staticmethod
    def create_high_dim_tiff(save_path, *images):
        """
        【高级功能】: 用于创建超过 4 通道的 TIFF 图像。
        既然你的任务涉及 >4 通道，你可能需要将多个图像（比如 RGB + 深度图 + 掩码）
        堆叠在一起保存为一个多通道 TIFF。
        
        :param save_path: 保存路径
        :param images: 多个同尺寸的 numpy 数组 (H, W, C)
        """
        try:
            # 在通道维度 (axis=2) 上进行堆叠
            # 例如：img1 (H,W,3) + img2 (H,W,3) -> stacked (H,W,6)
            stacked_img = np.dstack(images)
            
            print(f"[Info] 生成多通道图像，形状: {stacked_img.shape}")
            
            # OpenCV 的 imwrite 支持保存多通道 TIFF
            cv2.imwrite(str(save_path), stacked_img)
            return True
        except Exception as e:
            print(f"[Error] 保存多通道 TIFF 失败: {e}")
            return False

# ================= 使用示例 =================

if __name__ == "__main__":
    # 配置路径
    # data/raw_data/zky/jpeg
    # input_folder = os.path.join("data", "raw_data", "zky", "jpeg") 
    # output_folder = os.path.join("data", "zky", "tiff_imgs")

    # data\raw_data\hyper_kvasir\images
    input_folder = os.path.join("data", "raw_data", "hyper_kvasir", "images") 
    output_folder = os.path.join("data", "hyper_kvasir", "images")


    # 1. 实例化类
    processor = TiffPreprocessor(input_folder, output_folder)
    
    # 2. 运行批量转换
    # 这会将现有的 JPG/PNG (3或4通道) 转为 TIFF 格式
    processor.convert_to_tiff()

    # ========================================================
    # 场景模拟：如何利用此类处理 >4 通道的情况
    # ========================================================
    # 假设你以后想把两张处理好的 TIFF 合并成一个 6 通道的图像用于加密输入：
    
    # img_part1 = cv2.imread("data/tiff_preprocessed/image_01.tiff", cv2.IMREAD_UNCHANGED) # BGR (3通道)
    # img_part2 = cv2.imread("data/tiff_preprocessed/image_02.tiff", cv2.IMREAD_UNCHANGED) # BGR (3通道)
    
    # if img_part1 is not None and img_part2 is not None:
    #     TiffPreprocessor.create_high_dim_tiff("data/tiff_preprocessed/combined_6ch.tiff", img_part1, img_part2)
    #     print("6通道图像已保存。")