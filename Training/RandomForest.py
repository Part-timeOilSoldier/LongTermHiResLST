import os
import joblib
from sklearn.ensemble import RandomForestRegressor

from LST_Tools import Tool1, Tool2


def main():
    # 导入需要使用的数据
    x_train, x_test, y_train, y_test, model_path, picture_path = Tool2.load_and_preprocess_data()

    # 设置模型保存/调用路径
    model_filename = os.path.join(model_path, 'random_forest_model.joblib')

    # 判断模型是否已经存在
    if os.path.exists(model_filename):
        # 存在则直接进行加载
        regressor = joblib.load(model_filename)
    else:
        # 不存在则设置参数进行训练并保存
        regressor = RandomForestRegressor(n_estimators=200,
                                          max_depth=15,
                                          min_samples_split=5,
                                          min_samples_leaf=2,
                                          random_state=0,
                                          n_jobs=-1)
        regressor.fit(x_train, y_train)
        joblib.dump(regressor, model_filename)

    # 模型预测
    y_pred = regressor.predict(x_test)

    # 计算并打印模型精度
    Tool1.model_accuracy(y_test, y_pred, 3, 'random_forest_model', picture_path)


if __name__ == '__main__':
    main()
