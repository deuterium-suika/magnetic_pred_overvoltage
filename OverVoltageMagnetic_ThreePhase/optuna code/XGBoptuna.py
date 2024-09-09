# 支持向量回归
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.trial import TrialState
import sklearn
from xgboost import XGBRegressor

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

    n_estimator = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0, 1)
    gamma = trial.suggest_categorical("gamma", ["0.0", "0.1", "0.2", "0.3", "0.4"])
    reg_alpha = trial.suggest_float("reg_alpha", 0, 1)
    reg_lambda = trial.suggest_categorical("reg_lambda",["0", "0.1", "0.5", "1"])
    random_state =  trial.suggest_int("random_state", 1, 100)
    subsample = trial.suggest_float("subsample", 0.5, 0.9)
    # objective = trial.suggest_categorical("objective", ["reg:linear"])
    xgb = MultiOutputRegressor(XGBRegressor( n_estimators = n_estimator,
                                max_depth = max_depth,
                                learning_rate = learning_rate,
                                gamma = gamma,
                                reg_alpha = reg_alpha,
                                reg_lambda = reg_lambda,
                                random_state = random_state,
                                booster = 'gbtree',
                                eval_metric = 'rmse',
                                subsample = subsample,
                                objective = 'reg:squarederror'
    ))
    return xgb


def objective(trial):
    train_input = np.loadtxt('../data/zstrainInput.txt')  # (40, 1)
    # print(train_input.shape)
    test_input = np.loadtxt('../data/zstestInput.txt')
    train_output = np.loadtxt('../data/trainPCA.txt')
    # print(train_output.shape)
    test_output = np.loadtxt('../data/testPCA.txt')
    test_outreal = np.loadtxt('../data/testOutput.txt')

    # C值越大，模型越复杂
    xgb = define_model(trial)
    xgb.fit(train_input, train_output)

    y_pred = xgb.predict(test_input)
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
    study.optimize(objective, n_trials=50, timeout=60 * 200)  # 时间秒s
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
    with open('../optuna result/XGboost/BestTrial{}.txt'.format(trial.number), "a") as f:
        f.write("\n" + "MAEloss  " + detail)
        f.close()
