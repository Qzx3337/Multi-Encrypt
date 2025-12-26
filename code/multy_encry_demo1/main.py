import os
import shutil
import filecmp
import sys
import glob
from typing import Callable, Tuple, List

# ==============================================================================
# [SECTION 1] 用户算法接口绑定区 (User Algorithm Binding)
# 请在这里导入你写好的模块，例如: import my_stego, my_crypto
# ==============================================================================

def call_user_embed_algorithm(cover_path: str, secret_path: str, output_path: str):
    """
    [接口建议]: (src_img_path, src_txt_path, output_img_path) -> None
    请在此处调用你的嵌入函数。
    """
    # 示例: my_stego.embed(cover_path, secret_path, output_path)
    print(f"  [Mock Call] Calling USER EMBED algorithm...")
    print(f"    Input: {os.path.basename(cover_path)} + {os.path.basename(secret_path)}")
    
    # TODO: 删除下面这行，替换为你的实际调用
    # 为了防止直接运行报错，这里只是简单的复制文件模拟生成了文件，实际请务必替换
    shutil.copy2(cover_path, output_path) 
    # raise NotImplementedError("请在代码中连接你的 Embed 算法！")

def call_user_extract_algorithm(stego_path: str, output_path: str):
    """
    [接口建议]: (stego_img_path, output_txt_path) -> None
    请在此处调用你的提取函数。
    """
    # 示例: my_stego.extract(stego_path, output_path)
    print(f"  [Mock Call] Calling USER EXTRACT algorithm...")
    
    # TODO: 删除下面这行，替换为你的实际调用
    # 这里的模拟仅仅是为了让流程跑通，实际必须替换
    with open(output_path, 'w') as f: f.write("This is a Top Secret Message for testing flow.") 
    # raise NotImplementedError("请在代码中连接你的 Extract 算法！")

def call_user_encrypt_algorithm(input_path: str, output_path: str):
    """
    [接口建议]: (input_img_path, output_img_path) -> None
    请在此处调用你的加密函数。
    """
    # 示例: my_crypto.encrypt(input_path, output_path, key="secret")
    print(f"  [Mock Call] Calling USER ENCRYPT algorithm...")
    
    # TODO: 删除下面这行，替换为你的实际调用
    shutil.copy2(input_path, output_path)
    # raise NotImplementedError("请在代码中连接你的 Encrypt 算法！")

def call_user_decrypt_algorithm(input_path: str, output_path: str):
    """
    [接口建议]: (input_img_path, output_img_path) -> None
    请在此处调用你的解密函数。
    """
    # 示例: my_crypto.decrypt(input_path, output_path, key="secret")
    print(f"  [Mock Call] Calling USER DECRYPT algorithm...")
    
    # TODO: 删除下面这行，替换为你的实际调用
    shutil.copy2(input_path, output_path)
    # raise NotImplementedError("请在代码中连接你的 Decrypt 算法！")

# ==============================================================================
# [SECTION 2] 测试框架核心 (Test Harness Core)
# 负责环境搭建、流程控制、断言验证，不包含具体算法逻辑
# ==============================================================================

class TransmissionTestFramework:
    def __init__(self, base_dir="test_workspace"):
        self.base_dir = base_dir
        # 定义标准化的测试目录结构
        self.dirs = {
            "0_src": os.path.join(base_dir, "0_src"),
            "1_embedded": os.path.join(base_dir, "1_embedded"),
            "2_encrypted": os.path.join(base_dir, "2_encrypted"),
            "3_received": os.path.join(base_dir, "3_received"),
            "4_decrypted": os.path.join(base_dir, "4_decrypted"),
            "5_result": os.path.join(base_dir, "5_result")
        }

    def setup_environment(self):
        """
        初始化工作目录。
        注意：修改了逻辑，只会清理输出目录，**不会删除 0_src 中的源文件**。
        """
        # 仅清理生成的输出目录
        output_keys = ["1_embedded", "2_encrypted", "3_received", "4_decrypted", "5_result"]
        for key in output_keys:
            path = self.dirs[key]
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        
        # 确认源文件目录存在
        if not os.path.exists(self.dirs["0_src"]):
            os.makedirs(self.dirs["0_src"])
            print(f"⚠️ [Init] 源目录 {self.dirs['0_src']} 不存在，已自动创建。请将 d01.png, d01.txt 等文件放入此处。")
        else:
            print(f"✅ [Init] 环境已准备，源文件目录保留: {self.dirs['0_src']}")

    def load_test_assets(self) -> List[Tuple[str, str]]:
        """
        扫描 0_src 目录，寻找配对的测试文件。
        匹配规则：相同主文件名，分别以 .png 和 .txt 结尾。
        例如：(d01.png, d01.txt)
        """
        src_dir = self.dirs["0_src"]
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"源文件目录不存在: {src_dir}")

        # 寻找所有 png 文件
        png_files = glob.glob(os.path.join(src_dir, "*.png"))
        if not png_files:
            raise FileNotFoundError(f"在 {src_dir} 中未找到 .png 文件。请放入测试文件。")

        asset_pairs = []
        print(f"📂 [Loader] 正在扫描源文件目录: {src_dir}")
        
        for png_path in png_files:
            # 构建对应的 txt 路径 (d01.png -> d01.txt)
            base_name = os.path.splitext(os.path.basename(png_path))[0]
            txt_path = os.path.join(src_dir, f"{base_name}.txt")
            
            if os.path.exists(txt_path):
                asset_pairs.append((png_path, txt_path))
                print(f"  -> 发现配对: {base_name}.png + {base_name}.txt")
            else:
                print(f"  -> ⚠️ 跳过: 找到 {base_name}.png 但缺失对应的 .txt 文件")

        if not asset_pairs:
            raise ValueError("未找到任何完整的测试对 (同时拥有 .png 和 .txt)。")
        
        return asset_pairs

    def _verify_file_exists(self, path: str, step_name: str):
        """内部辅助：确保步骤生成了文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ [Error] 步骤 '{step_name}' 失败: 未生成输出文件 {path}")
        if os.path.getsize(path) == 0:
            print(f"⚠️ [Warning] 步骤 '{step_name}' 生成的文件为空: {path}")

    def check_ecorrectness(self, original: str, target: str, label: str) -> bool:
        """比对文件一致性 (E-correctness)"""
        print(f"🔎 [Check] 正在验证 {label}...")
        if not os.path.exists(original) or not os.path.exists(target):
            print(f"  -> ❌ 失败: 文件缺失")
            return False
            
        is_same = filecmp.cmp(original, target, shallow=False)
        if is_same:
            print(f"  -> ✅ PASS: 文件完全一致")
        else:
            print(f"  -> ❌ FAIL: 文件不一致")
        return is_same

    def simulate_transmission_channel(self, src_path: str, dest_dir: str) -> str:
        """模拟传输过程（网络传输、拷贝等）"""
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        # 可以在这里增加模拟丢包或噪声的逻辑来测试鲁棒性
        shutil.copy2(src_path, dest_path)
        return dest_path

    # ==========================================================================
    # [SECTION 3] 主流程控制 (Main Workflow)
    # 严格遵循: Src -> Embed -> Encrypt -> Trans -> Decrypt -> Check -> Extract -> Check
    # ==========================================================================
    
    def run_main(self):
        print("\n=== 启动双模加密传输测试流程 ===\n")
        
        # 1. 环境准备 (保留源文件)
        self.setup_environment()
        
        try:
            # 2. 读取所有测试文件对
            test_pairs = self.load_test_assets()
            
            # 3. 循环测试每一对
            for i, (src_img, src_txt) in enumerate(test_pairs):
                base_name = os.path.splitext(os.path.basename(src_img))[0]
                print(f"\n{'='*20} 开始测试组 {i+1}: {base_name} {'='*20}")
                
                try:
                    # -------------------------------------------------
                    # Step 1: Embed (Src -> Embedded)
                    # -------------------------------------------------
                    print("\n--- [Step 1] Embedding ---")
                    embedded_img = os.path.join(self.dirs["1_embedded"], f"{base_name}_embedded.png")
                    call_user_embed_algorithm(src_img, src_txt, embedded_img)
                    self._verify_file_exists(embedded_img, "Embedding")

                    # -------------------------------------------------
                    # Step 2: Encrypt (Embedded -> Encrypted)
                    # -------------------------------------------------
                    print("\n--- [Step 2] Encryption ---")
                    encrypted_img = os.path.join(self.dirs["2_encrypted"], f"{base_name}_encrypted.png")
                    call_user_encrypt_algorithm(embedded_img, encrypted_img)
                    self._verify_file_exists(encrypted_img, "Encryption")

                    # -------------------------------------------------
                    # Step 3: Transmission (Encrypted -> Received)
                    # -------------------------------------------------
                    print("\n--- [Step 3] Transmission ---")
                    received_img = self.simulate_transmission_channel(encrypted_img, self.dirs["3_received"])
                    self._verify_file_exists(received_img, "Transmission")

                    # -------------------------------------------------
                    # Step 4: Decrypt (Received -> Decrypted)
                    # -------------------------------------------------
                    print("\n--- [Step 4] Decryption ---")
                    decrypted_img = os.path.join(self.dirs["4_decrypted"], f"{base_name}_decrypted.png")
                    call_user_decrypt_algorithm(received_img, decrypted_img)
                    self._verify_file_exists(decrypted_img, "Decryption")

                    # -------------------------------------------------
                    # Check 1: Decryption E-correctness
                    # -------------------------------------------------
                    decry_success = self.check_ecorrectness(embedded_img, decrypted_img, "解密完整性 (Embedded vs Decrypted)")
                    if not decry_success:
                        print("⛔ [Stop] 解密校验失败，跳过本组后续步骤。")
                        continue # 跳过本组，继续下一组

                    # -------------------------------------------------
                    # Step 5: Extract (Decrypted -> Result)
                    # -------------------------------------------------
                    print("\n--- [Step 5] Extraction ---")
                    extracted_txt = os.path.join(self.dirs["5_result"], f"extracted_{base_name}.txt")
                    call_user_extract_algorithm(decrypted_img, extracted_txt)
                    self._verify_file_exists(extracted_txt, "Extraction")

                    # -------------------------------------------------
                    # Check 2: Final Data E-correctness
                    # -------------------------------------------------
                    final_success = self.check_ecorrectness(src_txt, extracted_txt, "最终提取数据 (Src Text vs Result Text)")

                    print(f"\n✅ [Result] 组 {base_name} 测试完成: {'全部通过' if final_success else '数据提取不一致'}")
                
                except Exception as task_e:
                    print(f"\n❌ [Error] 组 {base_name} 测试中发生错误: {task_e}")
                    import traceback
                    traceback.print_exc()
                    continue # 继续下一组

        except Exception as e:
            print(f"\n❌ [Critical Error] 全局流程异常:\n{e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_suite = TransmissionTestFramework()
    test_suite.run_main()