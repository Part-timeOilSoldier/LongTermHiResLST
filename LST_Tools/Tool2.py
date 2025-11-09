import glob
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

from LST_Tools import Tool1


def path_set():
    # 根据需要的具体参数进行调整输入输出路径
    x1_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m'
    x2_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m'
    x3_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m'
    y_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
    QA_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/QA_30m'

    model_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Models'
    picture_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/PerformanceMetrics'

    return x1_path, x2_path, x3_path, y_path, QA_path, model_path, picture_path


def predict_path_set():
    input_paths = ['F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m/',
                   'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m/',
                   'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/']
    output_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/LST/'
    QA_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/QA_30m/'
    return input_paths, output_path, QA_path


def filter_data_based_on_qa(x1, x2, x3, y, QA):
    """
    根据QA数据过滤x1, x2, x3, y数组，删除对应于QA值为0的条目。

    :param x1: 一维数组, 对应某个特定的特征或波段数据。
    :param x2: 一维数组, 对应另一个特定的特征或波段数据。
    :param x3: 一维数组, 对应第三个特定的特征或波段数据。
    :param y: 一维数组, 目标变量。
    :param QA: 一维数组, 质量保证/质量控制数据，用于指示相应位置的数据是否有效。
    :return: 过滤后的x1, x2, x3, y数组。
    """
    # 使用QA数组创建一个布尔索引，其中QA值为1的位置为True
    valid_indices = QA == 1

    # 应用布尔索引过滤数据
    x1_filtered = x1[valid_indices]
    x2_filtered = x2[valid_indices]
    x3_filtered = x3[valid_indices]
    y_filtered = y[valid_indices]

    return x1_filtered, x2_filtered, x3_filtered, y_filtered


def read_tiff_data(paths, i, clip_min=None, clip_max=None):
    # 确保索引i在paths范围内
    if i < 0 or i >= len(paths):
        raise IndexError("Index i is out of bounds for the paths list.")

    path = paths[i]  # 使用索引i选择特定路径
    vectorized_data = []

    # 只处理单个文件
    with rasterio.open(path) as src:
        data = src.read(1)  # 读取第一个波段

    # 数据处理部分
    if clip_min is not None or clip_max is not None:
        mean_value = np.mean(data)  # 计算平均值

        if clip_min is not None:
            data[data < clip_min] = mean_value  # 替换小于 clip_min 的值
        if clip_max is not None:
            data[data > clip_max] = mean_value  # 替换大于 clip_max 的值

    vectorized_data.append(data.ravel())

    return np.concatenate(vectorized_data)


def load_and_preprocess_data():
    # 设置数据路径
    x1_path, x2_path, x3_path, y_path, QA_path, model_path, picture_path = path_set()
    # 这里的QA数据不参与使用

    # 载入并预处理数据
    x1 = Tool1.preprocess_single_data(x1_path, 5, -5, 10)  # NDVI
    x2 = Tool1.preprocess_single_data(x2_path, 5, 100, 1000)  # LST_990m
    x3 = Tool1.preprocess_single_data(x3_path, 5)  # 土地利用类型30m
    y = Tool1.preprocess_single_data(y_path, 5, 100, 1000)  # LST_30m

    # 组合 x1, x2, x3 为训练数据
    X = np.vstack((x1, x2, x3)).T

    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=66)

    return x_train, x_test, y_train, y_test, model_path, picture_path


def load_data_usable(i):
    # 设置数据路径
    x1_path, x2_path, x3_path, y_path, QA_path, model_path, picture_path = path_set()

    # 载入并预处理数据
    x1 = read_tiff_data(glob.glob(f'{x1_path}/*.tif'), i, -5, 10)  # NDVI
    x2 = read_tiff_data(glob.glob(f'{x2_path}/*.tif'), i, 100, 1000)  # LST_990m
    x3 = read_tiff_data(glob.glob(f'{x3_path}/*.tif'), i)  # 土地利用类型30m
    y = read_tiff_data(glob.glob(f'{y_path}/*.tif'), i, 100, 1000)  # LST_30m
    QA = read_tiff_data(glob.glob(f'{QA_path}/*.tif'), i)  # 云覆盖数据

    # 使用QA数据过滤x1, x2, x3, y
    x1, x2, x3, y = filter_data_based_on_qa(x1, x2, x3, y, QA)

    # 组合 x1, x2, x3 为训练数据
    X = np.vstack((x1, x2, x3)).T
    # 检查 X 中的 NaN 和无穷大值
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        # 对每一列分别计算中位数
        median_values = np.array([np.median(X[~np.isinf(X[:, col]), col]) for col in range(X.shape[1])])

        # 替换无穷大值
        inf_positions = np.where(np.isinf(X))
        for pos in zip(*inf_positions):
            # 替换为对应列的中位数
            X[pos] = median_values[pos[1]]

    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=66)

    return x_train, x_test, y_train, y_test, model_path, picture_path
