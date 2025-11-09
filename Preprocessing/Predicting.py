import os

from LST_Tools import Tool1, Tool2


def main():
    for i in range(0, 12):

        model_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Models/2022'

        # 设置模型保存/调用路径
        model_filename = os.path.join(model_path, f'random_forest_model_{i + 1:02}.joblib')

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
