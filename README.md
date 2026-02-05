# Multi-Encrypt: 基于混沌系统的多模态内容自适应加密算法

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

</div>

## 项目简介 (Introduction)

本项目实现了一种改进的高级混沌加密算法。该工作在 Jin. ST 的研究基础上进行了多项关键改进，旨在解决传统图像加密算法在多模态数据适应性、密钥安全性以及抗攻击能力上的不足。

本算法不仅仅适用于单一图像，通过引入**多模态（图像+文本）**加密机制，结合**内容自适应的动态会话密钥**生成策略，显著提升了系统的安全层级和明文敏感性。

## 核心贡献 (Key Contributions)

相比于基线方案，本项目的改进主要体现在以下四个方面：

1.  **多模态支持改进**：将原本仅适用于图像的加密算法扩展为多模态加密，支持“图像+文字”作为被加密对象。
2.  **动态安全层级 (Content-Adaptive)**：
    * 摒弃了静态密钥/内置密钥 ($C = \text{Enc}(P)$) 模式。
    * 引入**动态会话密钥**机制：利用明文特征作为“动态盐值 (Salt)”，与用户口令混合生成密钥。
    * 实现了**一次一密 (One-Time Pad effect)**，具备极强的前向安全性。
3.  **增强的混沌初值生成**：改进了混沌序列的初值生成方式，大幅提升了算法对明文和密钥微小变化的敏感性。
4.  **三阶段加密架构**：将原有的二阶段通道内加密改进为 **“通道内置乱 -> 通道间扩散 -> DNA混沌编码”** 的三阶段架构，引入了通道间的信息交互。

## 算法流程 (Algorithm Overview)

本算法采用三阶段处理流程，确保了极高的混乱度（Confusion）和扩散性（Diffusion）。

### 阶段一：基于 Logistic 序列的 Q 掩模置乱
这是一个内容自适应的密钥派生与初步混淆过程。
* **密钥派生**：通过哈希函数将“用户密码”与“明文特征”进行数学耦合。
    $$K_{Init} = \mathcal{M}(\text{SHA-256}(K_{user} \oplus \mathcal{H}(Plain)))$$
    * 其中 $\mathcal{H}(Plain)$ 采用了基于**位置索引加权**的取模运算，确保任何像素位置或通道的变化都会导致生成的 Hash 值发生雪崩效应。
* **Q 掩模生成**：利用 $K_{Init}$ 驱动 Logistic 混沌映射生成掩码矩阵 $Q$。
* **初步混淆**：$I' = I \oplus Q$，对图像进行按位异或，平坦化直方图。

### 阶段二：基于分块循环的通道间扩散
为了打破色彩通道间的独立性，引入通道间扩散机制。
* 从第 $N$ 通道开始向前异或直到第 1 通道，使得任一通道的信息泄露都依赖于其他通道，增加了破解难度。

### 阶段三：基于同步混沌系统的 DNA 编解码
利用四维同步混沌系统与 DNA 编码规则进行深度加密。
* **4D 混沌系统**：生成包含四个分量的长混沌序列 $S^{(1)}, S^{(2)}, S^{(3)}, S^{(4)}$。
* **DNA 变换**：将扩散后的数据分割为 $P \times P$ 的非重叠块 ($B_k$)，进行四轮 DNA 编解码变换：
    $$B_k' = \Omega^{-1}_{s^{(4)}_k} \left( \Omega_{s^{(3)}_k} \left( \Omega^{-1}_{s^{(2)}_k} \left( \Omega_{s^{(1)}_k} (B_k) \right) \right) \right)$$
    * $\Omega$：二进制转 DNA 序列。
    * $\Omega^{-1}$：DNA 序列转二进制。
    * 利用混沌序列动态选择 DNA 编码/解码规则（如规则 R1-R8）。

## 文件结构 (File Structure)

```text
multi-encrypt/
├── src/
│   ├── multy_encry/           # 核心加密算法实现
│   │   ├── main.py            # 主程序入口
│   │   ├── chaos_encryptor.py # 加密器类封装
│   │   ├── chaos.py           # 混沌系统定义
│   │   └── ...
│   ├── text_embed/            # 多模态处理（文本嵌入/提取）
│   │   ├── s1_embed.py        # 文本嵌入逻辑
│   │   ├── s2_extract.py      # 文本提取逻辑
│   │   └── ...
│   ├── utils/                 # 工具脚本
│   │   ├── analyze_correlations.py # 相关性分析
│   │   ├── draw_histogram_*.py     # 直方图绘制工具
│   │   └── preprocess.py           # 数据预处理
│   └── DRL/                   # 深度强化学习相关模块 (Base/Experimental)
├── experiments/               # 实验数据与结果
├── requirements.txt           # 项目依赖
└── .gitignore
```

## 引用 (References)
本工作基于 Jin. ST 等人的基础研究改进。

Created by [qzx]
