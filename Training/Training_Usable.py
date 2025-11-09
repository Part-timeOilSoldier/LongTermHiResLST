import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from LST_Tools import Tool1, Tool2


def main():
    for i in range(0, 12):
        # 导入需要使用的数据
        x_train, x_test, y_train, y_test, model_path, picture_path = Tool2.load_data_usable(i)
        # 这里返回的已经是去除云数据后的训练数组

        model_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Models/2022'

        # 设置模型保存/调用路径
        model_filename = os.path.join(model_path, f'random_forest_model_{i + 1:02}.joblib')

        # 检查模型文件是否已存在，如果存在，则跳过训练
        if os.path.exists(model_filename):
            # print(f"模型 {model_filename} 已存在")
            regressor = joblib.load(model_filename)
        else:
            # 设置模型训练参数
            regressor = RandomForestRegressor(n_estimators=200,
                                              max_depth=15,
                                              min_samples_split=5,
                                              min_samples_leaf=2,
                                              random_state=0,
                                              n_jobs=7)

            # 模型训练
            regressor.fit(x_train, y_train)
            # 保存模型文件
            joblib.dump(regressor, model_filename)

        # 模型预测
        y_pred = regressor.predict(x_test)

        # 计算R²值并打印R²值
        r2 = r2_score(y_test, y_pred)
        print(f"R²: {r2}")

        # 准备predict_and_save函数的参数
        input_paths, output_path, base_qa_path = Tool2.predict_path_set()
        input_paths = [os.path.join(path, sorted(os.listdir(path))[i]) for path in input_paths]
        output_path = os.path.join(output_path, f'Predict_LST_{i + 1:02}.tif')
        QA_path = os.path.join(base_qa_path, sorted(os.listdir(base_qa_path))[i])

        # 检查预测结果文件是否已存在，若存在则跳过
        if not os.path.exists(output_path):
            # 文件不存在，调用predict_and_save函数
            Tool1.predict_and_save(model_filename, input_paths, output_path, QA_path)
        else:
            print(f"图像 {output_path} 已存在")


if __name__ == '__main__':
    main()
