import joblib
import rasterio
import numpy as np
from sklearn.metrics import r2_score


# 修改后的 predict_path_set 函数
def predict_path_set():
    input_paths = {
        'ndvi': 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/NDVI30/NDVI_20220617.tif',
        'lst': 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LST30990/LST_Day_1km_20220616.tif',
        'type': 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LCD30/jiangsu.tif',
        'true_lst': 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LST30/LC08_L2SP_120038_20220617_20220629_02_T1_ST_B10_processed.tif'
    }
    output_path = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/Pre30/predict_lst.tif'
    return input_paths, output_path


# 修改后的 predict_and_save 函数
def predict_and_save(model_path, input_paths, output_path):
    # 加载模型
    model = joblib.load(model_path)

    # 读取 NDVI、LST、Type 和真实的 LST 数据
    ndvi = rasterio.open(input_paths['ndvi']).read(1)
    lst = rasterio.open(input_paths['lst']).read(1)
    type_data = rasterio.open(input_paths['type']).read(1)
    true_lst = rasterio.open(input_paths['true_lst']).read(1)

    # 检查并调整形状以确保一致性
    if ndvi.shape != lst.shape or ndvi.shape != type_data.shape or ndvi.shape != true_lst.shape:
        min_shape = np.min([ndvi.shape, lst.shape, type_data.shape, true_lst.shape], axis=0)
        ndvi = ndvi[:min_shape[0], :min_shape[1]]
        lst = lst[:min_shape[0], :min_shape[1]]
        type_data = type_data[:min_shape[0], :min_shape[1]]
        true_lst = true_lst[:min_shape[0], :min_shape[1]]

    # 将输入数据堆叠为 (height, width, bands)
    X = np.stack([ndvi, lst, type_data], axis=-1)
    X_reshaped = X.reshape(-1, X.shape[-1])  # 重塑为 (n_samples, n_features)
    true_lst_reshaped = true_lst.reshape(-1)

    # 检查 X 中的无穷大值
    inf_count = np.isinf(X_reshaped).sum()
    if inf_count > 0:
        # 使用 X_reshaped.shape[1] 确保我们获取正确列数
        median_values = np.array(
            [np.median(X_reshaped[~np.isinf(X_reshaped[:, col]), col]) for col in range(X_reshaped.shape[1])]
        )

        # 替换无穷大值
        inf_positions = np.where(np.isinf(X_reshaped))
        for pos in zip(*inf_positions):
            # 替换为对应列的中位数
            X_reshaped[pos] = median_values[pos[1]]

    y_pred = model.predict(X_reshaped)
    # 将预测结果填充回原始形状
    y_pred_reshaped = y_pred.reshape(X.shape[:-1])  # 二维重塑

    # 计算 R² 值
    r2 = r2_score(true_lst_reshaped, y_pred)
    print(f'R² 值: {r2}')

    # 保存预测结果
    with rasterio.open(input_paths['lst']) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(y_pred_reshaped.astype(rasterio.float32), 1)
            print(f'预测结果已保存到 {output_path}')


# 主程序
model_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Models/2022/random_forest_model_06.joblib'  # 单个模型文件路径

# 设置预测路径
input_paths, output_path = predict_path_set()

# 调用预测函数
predict_and_save(model_path, input_paths, output_path)

print("完成预测")
