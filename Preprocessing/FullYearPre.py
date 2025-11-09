import os
import joblib
import rasterio
import numpy as np


# 修改后的 predict_path_set 函数
def predict_path_set(month):
    lst_month_path = os.path.join('F:/MyProjects/MachineLearning/Data/FullYearPre/TempData/LSTmouth', f'{month:02}')
    input_paths = [
        'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m/',
        lst_month_path,
        'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/'
    ]
    output_path = 'F:/MyProjects/MachineLearning/Data/FullYearPre/FinalData'
    qa_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/QA_30m/'
    return input_paths, output_path, qa_folder


# 修改后的 predict_and_save 函数
def predict_and_save(model_path, input_paths, output_path, qa_folder, month):
    # 加载模型
    model = joblib.load(model_path)

    # 获取输入数据文件夹路径
    ndvi_folder, lst_month_path, type_folder = input_paths

    # 获取该月 NDVI、Type 和 QA 数据文件（假设每月一个文件）
    ndvi_path = sorted(os.listdir(ndvi_folder))[month - 1]
    ndvi_path = os.path.join(ndvi_folder, ndvi_path)
    type_path = sorted(os.listdir(type_folder))[month - 1]
    type_path = os.path.join(type_folder, type_path)
    qa_path = sorted(os.listdir(qa_folder))[month - 1]
    qa_path = os.path.join(qa_folder, qa_path)

    # 读取 NDVI、Type 和 QA 数据
    ndvi = rasterio.open(ndvi_path).read(1)
    type_data = rasterio.open(type_path).read(1)
    qa_data = rasterio.open(qa_path).read(1).flatten()  # 确保 QA 数据与 X 的形状一致

    for lst_filename in sorted(os.listdir(lst_month_path)):
        lst_path = os.path.join(lst_month_path, lst_filename)
        lst = rasterio.open(lst_path).read(1)

        # 将输入数据堆叠为 (height, width, bands)
        X = np.stack([ndvi, lst, type_data], axis=-1)
        X_reshaped = X.reshape(-1, X.shape[-1])  # 重塑为 (n_samples, n_features)

        # 应用 QA 数据过滤
        if X_reshaped.shape[0] != qa_data.size:
            raise ValueError("QA 数据与输入数据尺寸不匹配")
        valid_indices = qa_data == 1
        X_filtered = X_reshaped[valid_indices]

        # 检查X中的无穷大值
        inf_count = np.isinf(X_filtered).sum()
        if inf_count > 0:
            # 使用 X_reshaped.shape[1] 确保我们获取正确列数
            median_values = np.array(
                [np.median(X_filtered[~np.isinf(X_filtered[:, col]), col]) for col in range(X_filtered.shape[1])])

            # 替换无穷大值
            inf_positions = np.where(np.isinf(X_filtered))
            for pos in zip(*inf_positions):
                # 替换为对应列的中位数
                X_filtered[pos] = median_values[pos[1]]

        y_pred = model.predict(X_filtered)
        # 将预测结果填充回原始形状
        y_pred_full = np.full(qa_data.shape, np.nan)  # 初始化全 NaN 数组
        y_pred_full[valid_indices] = y_pred  # 将预测结果填充到相应位置
        y_pred_reshaped = y_pred_full.reshape(X.shape[:-1])  # 二维重塑

        # 保存预测结果
        output_filename = f'Predict_LST_{os.path.splitext(lst_filename)[0]}.tif'
        with rasterio.open(lst_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)
            output_file = os.path.join(output_path, output_filename)
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(y_pred_reshaped.astype(rasterio.float32), 1)
                print(output_filename)


# 主程序
for i in range(1, 13):
    model_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Models/2022'  # 文件夹存储了每月的模型文件

    # 设置模型保存/调用路径
    model_filename = os.path.join(model_path, f'random_forest_model_{i:02}.joblib')

    # 准备predict_and_save函数的参数
    input_paths, output_path, qa_folder = predict_path_set(i)

    # 调用预测函数
    predict_and_save(model_filename, input_paths, output_path, qa_folder, i)

print("完成所有预测")
