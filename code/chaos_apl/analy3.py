import math
from collections import Counter


def calculate_entropy(string):
    """计算给定字符串的信息熵"""
    # 计算每个字符出现的频率
    frequency_list = Counter(string)

    # 总字符数
    total_length = len(string)

    # 计算信息熵
    entropy = -sum(
        count / total_length * math.log2(count / total_length)
        for count in frequency_list.values()
    )

    return entropy


# 示例用法
if __name__ == "__main__":
    test_string = "informationentropy"
    print("信息熵为: {calculate_entropy(test_string):.2f}")