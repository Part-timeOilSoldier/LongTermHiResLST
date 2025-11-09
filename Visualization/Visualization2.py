import rasterio
import numpy as np


def main():
    # 修改此处为实际的土地覆盖数据 tif 文件路径
    landcover_file = r"E:\MyProjects\MachineLearning\Data\Usable_Data\Type_30m\anhuiwgs84_6.tif"

    # 读取土地覆盖数据（假设为单波段数据）
    with rasterio.open(landcover_file) as src:
        landcover_data = src.read(1)

    # 如果 landcover_data 为 masked 数组，则填充掩膜值
    if np.ma.is_masked(landcover_data):
        landcover_data = landcover_data.filled()

    # 使用 numpy.unique 统计所有不同类别像元的数量，不忽略任何值
    categories, counts = np.unique(landcover_data, return_counts=True)
    total_pixels = landcover_data.size

    # 在控制台打印统计结果（占比保留 4 位小数）
    print("土地覆盖数据统计结果：")
    print("{:<10} {:<15} {:<12}".format("类别", "像元数量", "占比 (%)"))
    for cat, count in zip(categories, counts):
        percent = count / total_pixels * 100
        print("{:<10} {:<15} {:<12.4f}".format(cat, count, percent))


if __name__ == "__main__":
    main()
