import os
import joblib
from catboost import CatBoostRegressor

from LST_Tools import Tool1, Tool2


def main():
    # 导入需要使用的数据
    x_train, x_test, y_train, y_test, model_path, picture_path = Tool2.load_and_preprocess_data()

    # 设置模型保存/调用路径
    model_filename = os.path.join(model_path, 'catboost_model.joblib')

    # 判断模型是否已经存在
    if os.path.exists(model_filename):
        # 存在则直接进行加载
        clf = joblib.load(model_filename)
    else:
        # 不存在则设置参数进行训练并保存
        clf = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=500,
            l2_leaf_reg=3,
            random_state=0
        )
        clf.fit(x_train, y_train)
        joblib.dump(clf, model_filename)

    # 模型预测
    y_pred = clf.predict(x_test)

    # 计算并打印模型精度
    Tool1.model_accuracy(y_test, y_pred, 3, 'catboost_model', picture_path)


if __name__ == '__main__':
    main()
