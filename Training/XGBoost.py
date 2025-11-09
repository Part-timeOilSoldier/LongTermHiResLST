import os
import joblib
import xgboost as xgb
import cupy as cp  # 导入 cupy 用于 GPU 数据处理

from LST_Tools import Tool1, Tool2


def main():
    # 导入需要使用的数据
    x_train, x_test, y_train, y_test, model_path, picture_path = Tool2.load_and_preprocess_data()

    # 转移数据到 GPU
    x_train_gpu = cp.array(x_train)
    x_test_gpu = cp.array(x_test)
    y_train_gpu = cp.array(y_train)
    y_test_gpu = cp.array(y_test)

    # 设置模型保存/调用路径
    model_filename = os.path.join(model_path, 'xgboost_model.joblib')

    # 判断模型是否已经存在
    if os.path.exists(model_filename):
        # 存在则直接进行加载
        xgm = joblib.load(model_filename)
        xgm.set_params(device='cuda')  # 确保模型在 GPU 上
    else:
        # 不存在则设置参数进行训练并保存
        xgm = xgb.XGBRegressor(
            max_depth=10,
            learning_rate=0.1,
            n_estimators=300,
            reg_lambda=1,
            tree_method='hist',  # 使用直方图算法
            device='cuda',  # 指定使用 GPU
            random_state=66
        )
        xgm.fit(x_train_gpu, y_train_gpu, eval_set=[(x_test_gpu, y_test_gpu)], verbose=False)
        joblib.dump(xgm, model_filename)

    # 模型预测
    y_pred_gpu = xgm.predict(x_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu)  # 将预测结果转回 CPU

    # 计算并打印模型精度
    Tool1.model_accuracy(y_test, y_pred, 3, 'xgboost_model', picture_path)


if __name__ == '__main__':
    main()
