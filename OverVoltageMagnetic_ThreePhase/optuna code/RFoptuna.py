# 支持向量回归
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.trial import TrialState
import sklearn
from sklearn.ensemble import RandomForestRegressor

# 需要还原后计算mape
def mape_fn(yTrue, yPred):
    # for i in range(yTrue)
    return np.mean(np.abs((yPred - yTrue) / yTrue)) * 100


def rebuild_PCA(pca_data):
    # pca_data = np.loadtxt('../data/train_output.txt', encoding='utf-8', comments='%')
    mean_pca = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')

    # print(vector_pca.shape)  # (296277,)
    # print(pca_data.shape)  # (5, 1)
    huanyuan_data = np.matmul(pca_data, vector_pca) + mean_pca
    # huanyuan_data = np.matmul(pca_data.reshape(-1, 1), vector_pca.reshape(1, -1)) + mean_pca
    # print(raw_data.shape)
    # print(huanyuan_data.shape)
    # print('mape: ' + str(mape_fn(raw_data, huanyuan_data)) + '%')
    return huanyuan_data


def define_model(trial):

    n_estimator = trial.suggest_int("n_estimators", 1, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 200)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    max_feature = trial.suggest_categorical("max_features", ["auto", "log2", "sqrt"])
    min_samples_leaf = trial.suggest_int("min_samples_split", 2, 10)

    rf = RandomForestRegressor( n_estimators = n_estimator,
                                criterion = 'mse',
                                max_depth = max_depth,
                                min_samples_split = min_samples_split,
                                min_samples_leaf = min_samples_leaf,
                                # min_weight_fraction_leaf = 0.0,
                                max_features = max_feature,
                                # max_leaf_nodes = None,
                                # min_impurity_split = 1e-07,
                                # bootstrap = True,
                                # oob_score = False,
                                # n_jobs = 1,
                                # random_state = None,
                                # verbose = 0,
                                # warm_start = False
    )
    return rf


def objective(trial):
    train_input = np.loadtxt('../data/zstrainInput.txt')  # (40, 1)
    # print(train_input.shape)
    test_input = np.loadtxt('../data/zstestInput.txt')
    train_output = np.loadtxt('../data/trainPCA.txt')
    # print(train_output.shape)
    test_output = np.loadtxt('../data/testPCA.txt')
    test_outreal = np.loadtxt('../data/testOutput.txt')

    # C值越大，模型越复杂
    rf = define_model(trial)
    rf.fit(train_input, train_output)

    y_pred = rf.predict(test_input)
    pred_rebuild = rebuild_PCA(y_pred)

    # print("交叉检验R^2: ", np.mean(cross_val_score(rr_lin, train_input, train_output, cv=5)))
    # 回归系数a
    # print('回归系数a: ', svr_lin.coef_)
    # 截距b
    # print('截距b: ', svr_lin.intercept_)
    # MSN均方误差
    print('MSE: ', mean_squared_error(test_outreal, pred_rebuild))
    print('MAE: ', mean_absolute_error(test_outreal, pred_rebuild))
    print('rmse: ', np.sqrt(mean_squared_error(test_outreal, pred_rebuild)))
    # print('MAPE: ', mape_fn(test_outreal, pred_rebuild))
    print('r2: ' + str(r2_score(test_outreal, pred_rebuild)))
    print('MAPE: ', mape_fn(test_outreal, pred_rebuild))

    return mean_absolute_error(test_outreal, pred_rebuild)


if __name__ == "__main__":

    study = optuna.create_study(direction="minimize", study_name='net_trail')
    studyName = study.study_name
    study.optimize(objective, n_trials=100, timeout=60 * 200)  # 时间秒s
    """参数优化结束"""
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    """打印细节"""
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial: ", trial.number)
    print("  Value: ", trial.value)   # value就是objective中return的值，即loss

    print("  Params: ")
    detail = str(float(trial.value)) + "\n"

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        detail = detail + "{}\t{}".format(key, value) + "\n"

    # np.savetxt('./BestTrial{}.txt'.format(trial.number), detail)
    print('*********************************************************')
    print(detail)
    with open('../optuna result/RF/BestTrial{}.txt'.format(trial.number), "a") as f:
        f.write("\n" + "MAEloss  " + detail)
        f.close()
