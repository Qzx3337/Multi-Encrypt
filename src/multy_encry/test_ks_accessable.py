def print_bin(name, value):
    """辅助函数：打印变量名、十进制值和二进制形式"""
    # :08b 表示打印为8位二进制，不足补0
    print(f"[{name}] Val: {value:<3} | Bin: {value:08b}")

def run_demo():
    print("=== 1. 初始状态 (Initial State) ===")
    # 假设这是我们的 1x1x4 数据
    R = 170  # 二进制 10101010
    G = 240  # 二进制 11110000
    B = 15   # 二进制 00001111
    T = 51   # 二进制 00110011 (假设这是文本转成的数字)
    
    print_bin("R ", R)
    print_bin("G ", G)
    print_bin("B ", B)
    print_bin("T ", T)
    print("-" * 40)

    # ===============================
    # 加密过程 (Encryption)
    # ===============================
    print("=== 2. 加密过程 (开始混淆) ===")
    
    # 步骤 1: R 混入 G 的信息
    # R_new = R XOR G
    R = R ^ G
    print_bin("R'", R)
    
    # 步骤 2: G 混入 B 的信息
    # G_new = G XOR B
    G = G ^ B
    print_bin("G'", G)
    
    # 步骤 3: B 混入 T (文本) 的信息
    # B_new = B XOR T
    B = B ^ T
    print_bin("B'", B)
    
    # 步骤 4: T 混入 R (注意：此时的 R 已经是 R' 了！)
    # 这一步构成了闭环：文本不仅隐藏了自己，还混入了图像的信息
    T = T ^ R
    print_bin("T'", T)
    
    print(f"\n加密完成。密文数据: R={R}, G={G}, B={B}, T={T}")
    print("-" * 40)

    # ===============================
    # 解密过程 (Decryption)
    # 必须严格倒序！
    # ===============================
    print("=== 3. 解密过程 (倒序还原) ===")

    # 逆向步骤 4: 还原 T
    # 原理: (T' XOR R') = (T XOR R') XOR R' = T
    T = T ^ R
    print_bin("T (还原)", T)

    # 逆向步骤 3: 还原 B
    # 我们利用刚刚还原出来的 T 来解开 B
    B = B ^ T
    print_bin("B (还原)", B)

    # 逆向步骤 2: 还原 G
    # 利用还原出的 B 来解开 G
    G = G ^ B
    print_bin("G (还原)", G)

    # 逆向步骤 1: 还原 R
    # 利用还原出的 G 来解开 R
    R = R ^ G
    print_bin("R (还原)", R)

    print("-" * 40)
    print("演示结束：所有数据完美还原。")

if __name__ == "__main__":
    run_demo()
    