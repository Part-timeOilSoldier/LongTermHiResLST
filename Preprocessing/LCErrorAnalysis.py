import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rasterio
import os

# 假设数据存储在以下路径
land_use_type_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/anhuiwgs84_6.tif'
actual_lst_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m/LC08_L2SP_121038_20220616_20220411_02_T1_ST_B10_processed.tif'
predicted_lst_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Predict_Data/Predict_LST_06.tif'

# 定义一个函数来读取tif文件并将其展平为一维数组
def read_and_flatten_tif(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with rasterio.open(file_path) as src:
        array = src.read(1)  # 读取第一个波段
    return array.flatten()

try:
    # 加载数据并展平为一维数组
    land_use_type_flat = read_and_flatten_tif(land_use_type_path)
    actual_lst_flat = read_and_flatten_tif(actual_lst_path)
    predicted_lst_flat = read_and_flatten_tif(predicted_lst_path)

    # 重新分类，将4和7的值改为2
    land_use_type_flat[(land_use_type_flat == 4) | (land_use_type_flat == 7)] = 2

    # 创建DataFrame
    data = {
        'Land Use Type': land_use_type_flat,
        'Actual LST': actual_lst_flat,
        'Predicted LST': predicted_lst_flat
    }
    df = pd.DataFrame(data)

    # 计算误差
    df['Error'] = df['Predicted LST'] - df['Actual LST']

    # 将无限值转换为NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 定义分类字典，用于x轴标签和数据替换
    classification_dict = {
        1: "Farmland",
        2: "Forest",
        5: "Water",
        8: "Building"
    }

    # 替换数据中的土地利用类型数字为对应的名称
    df['Land Use Type'] = df['Land Use Type'].map(classification_dict)

    # 确保所有需要的类别都存在，即使没有数据也要显示
    all_types = ["Farmland", "Forest", "Water", "Building"]
    for land_use_type in all_types:
        if land_use_type not in df['Land Use Type'].unique():
            df = df.append(
                {'Land Use Type': land_use_type, 'Actual LST': np.nan, 'Predicted LST': np.nan, 'Error': np.nan},
                ignore_index=True)

    # 设置科研常用颜色的调色板
    colors = ["#5DADE2", "#F5B041", "#58D68D", "#AF7AC5"]
    sns.set_palette(sns.color_palette(colors))

    # 可视化不同土地利用类型的误差分布，不含异常点
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 18})
    sns.boxplot(x='Land Use Type', y='Error', data=df, order=all_types, width=0.5, linewidth=1.5, showfliers=False, palette=colors)
    plt.xlabel('', fontsize=20)
    plt.ylabel('Error (Predicted LST - Actual LST)', fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(-7.5, 7.5)

    # 添加图例，并移动到右边外部
    handles = [
        plt.Line2D([0], [0], color=color, lw=4, label=label)
        for color, label in zip(colors, all_types)
    ]
    plt.legend(handles=handles, title='Land Use Type', title_fontsize='13', fontsize='13', loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig('F:/论文/论文基础/图片/error_distribution_by_land_use_no_outliers.png', dpi=1000)
    plt.show()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"Error: {e}")
