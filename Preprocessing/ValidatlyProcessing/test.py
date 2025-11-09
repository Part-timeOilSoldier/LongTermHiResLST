import rasterio
import numpy as np
from collections import Counter


def list_unique_values(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # 读取第一波段的数据
        unique, counts = np.unique(data, return_counts=True)  # 获取唯一值及其计数
        value_counts = dict(zip(unique, counts))  # 将唯一值和计数组合成字典
    return value_counts


def main():
    tif_path = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LCD30/jiangsu.tif'  # 替换为你的TIF文件路径
    value_counts = list_unique_values(tif_path)

    print(f"唯一值及其计数：")
    for value, count in value_counts.items():
        print(f"值: {value}, 计数: {count}")


if __name__ == "__main__":
    main()
