import numpy as np

class InterleaveSplitter:
    def __init__(self, grid_size: int = 8):
        """
        Args:
            grid_size (int): 将宽和高分为多少份，默认为8。
                             生成的 Block 固定为 grid_size x grid_size。
        """
        self.grid = grid_size

    def split(self, matrix: np.ndarray) -> list:
        """
        将二维矩阵拆分为交织块。
        
        Args:
            matrix (np.ndarray): H x W 的图像矩阵
        Returns:
            blocks (list): 包含所有 grid_size x grid_size 小块的列表
        """
        h, w = matrix.shape
        
        # 1. 计算并执行 Padding (确保能被 grid_size 整除)
        pad_h = (self.grid - h % self.grid) % self.grid
        pad_w = (self.grid - w % self.grid) % self.grid
        
        matrix_pad = np.pad(matrix, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        h_pad, w_pad = matrix_pad.shape
        
        # 计算网格单元(Cell)尺寸
        cell_h = h_pad // self.grid
        cell_w = w_pad // self.grid

        # 2. 维度变换 (Vectorized)
        # Reshape: (grid, cell_h, grid, cell_w)
        reshaped = matrix_pad.reshape(self.grid, cell_h, self.grid, cell_w)

        # Transpose: 把 cell 维度移到前面 -> (cell_h, cell_w, grid, grid)
        # 这样前两个维度遍历所有 block，后两个维度构成 block 内容
        transposed = reshaped.transpose(1, 3, 0, 2)

        # 3. 展平为列表
        blocks_array = transposed.reshape(-1, self.grid, self.grid)
        
        # 转换为 list
        return [blocks_array[i] for i in range(blocks_array.shape[0])]

    def merge(self, blocks: list, original_height: int, original_width: int) -> np.ndarray:
        """
        根据原始尺寸将 blocks 恢复为原图。
        
        Args:
            blocks (list): split 函数生成的列表
            original_height (int): 原图高度 (用于计算 padding 和 裁剪)
            original_width (int): 原图宽度
            
        Returns:
            matrix (np.ndarray): 恢复后的图像
        """
        # 1. 根据传入的原始尺寸，重新计算 Padding 后的尺寸
        # 这是为了知道如何将 blocks 还原为大的矩阵
        pad_h = (self.grid - original_height % self.grid) % self.grid
        pad_w = (self.grid - original_width % self.grid) % self.grid
        
        h_pad = original_height + pad_h
        w_pad = original_width + pad_w
        
        cell_h = h_pad // self.grid
        cell_w = w_pad // self.grid

        # 校验 blocks 数量是否匹配 (可选，但推荐)
        expected_blocks = cell_h * cell_w
        if len(blocks) != expected_blocks:
            raise ValueError(f"Block 数量不匹配。期望 {expected_blocks}, 实际 {len(blocks)}。请检查传入的宽高是否正确。")

        # 2. 堆叠并逆向 Reshape
        blocks_array = np.array(blocks) # Shape: (N, grid, grid)
        
        # 还原为: (cell_h, cell_w, grid, grid)
        reshaped_back = blocks_array.reshape(cell_h, cell_w, self.grid, self.grid)

        # 3. 逆向 Transpose
        # 从 (cell_h, cell_w, grid, grid) 变回 (grid, cell_h, grid, cell_w)
        # 索引变换: 0->2, 1->3, 2->0, 3->1 => transpose(2, 0, 3, 1)
        transposed_back = reshaped_back.transpose(2, 0, 3, 1)

        # 4. 恢复大图并裁剪
        matrix_pad = transposed_back.reshape(h_pad, w_pad)
        
        # 使用传入的原始尺寸进行裁剪
        matrix = matrix_pad[:original_height, :original_width]

        return matrix

# ================= 验证流程 =================

if __name__ == "__main__":
    # 模拟场景：发送端
    # 1. 原始数据
    h_real, w_real = 10, 10
    original_img = np.arange(h_real * w_real).reshape(h_real, w_real)
    
    splitter = InterleaveSplitter(grid_size=8)
    
    # 2. 拆分 (发送端不需要知道接收端怎么处理，只需输出 blocks)
    blocks = splitter.split(original_img)
    print(f"发送端: 生成了 {len(blocks)} 个 blocks")

    # ----- 模拟传输 (仅传输 blocks, h_real, w_real) -----
    
    # 模拟场景：接收端
    # 3. 接收端已知：blocks 列表，以及 原图尺寸 (h_real, w_real)
    rec_h, rec_w = 10, 10 
    
    # 4. 合并 (显式传入尺寸)
    restored_img = splitter.merge(blocks, original_height=rec_h, original_width=rec_w)
    
    print(f"接收端: 恢复图尺寸 {restored_img.shape}")
    
    # 5. 校验
    if np.array_equal(original_img, restored_img):
        print("验证成功: 图像完美复原")
    else:
        print("验证失败: 图像数据不一致")
        
    # 测试一下如果尺寸传错了会怎样
    try:
        splitter.merge(blocks, 100, 100)
    except ValueError as e:
        print(f"\n错误捕获测试 (预期内): {e}")