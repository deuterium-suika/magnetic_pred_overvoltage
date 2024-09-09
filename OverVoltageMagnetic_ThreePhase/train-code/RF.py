# 用交叉验证与optuna超参数调优嵌套的RF训练及测试代码

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rebuildPCA(lowdim_data):
    # pca_data = np.loadtxt('../result/mix/LR/predY_linearmodel.txt', encoding='utf-8', comments='#')
    mean_pca = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')
    # print(lowdim_data.shape)    # (4,)
    # print(vector_pca.shape)    # (296277,)
    # print(mean_pca.shape)   # (296277,)
    huanyuan_data = np.matmul(lowdim_data, vector_pca) + mean_pca
    # np.savetxt('../result/DNN/predY_DNNmodel_rebuild.txt', huanyuan_data)
    return huanyuan_data


# 需要还原后计算mape
def mape_fn(yTrue, yPred):
    # for i in range(yTrue)
    return np.mean(np.abs((yPred - yTrue) / yTrue)) * 100


train_input = np.loadtxt('../data/zstrainInput.txt')
# print(train_input.shape)
test_input = np.loadtxt('../data/zstestInput.txt')
train_output = np.loadtxt('../data/trainPCA.txt')
# print(train_output.shape)
test_output = np.loadtxt('../data/testPCA.txt')
test_outreal = np.loadtxt('../data/testOutput.txt')


rf = RandomForestRegressor(n_estimators = 891,
                           criterion = 'mae',
                           max_depth = 135,
                                # min_samples_split = 2,
                                # min_samples_leaf = 5,
                                # min_weight_fraction_leaf = 0.0,
                           max_features = 'auto',
                                # max_leaf_nodes = None,
                                # min_impurity_split = 1e-07,
                                # bootstrap = True,
                                # oob_score = False,
                                # n_jobs = 1,
                                # random_state = None,
                                # verbose = 0,
                                # warm_start = False
)

rf.fit(train_input, train_output)
pred = rf.predict(test_input)
rebuildy_pre = rebuildPCA(pred)

test_mae = mean_absolute_error(test_outreal, pred)
test_mape = mape_fn(test_outreal, pred)
test_mse = mean_squared_error(test_outreal, pred)
test_r2 = r2_score(test_outreal, pred)

print('rmse: ', np.sqrt(mean_squared_error(test_outreal, pred)))
print('测试集MSE:' + str(test_mse))
print('测试集MAE:' + str(test_mae))
print('测试集MAPE:' + str(test_mape))
print('测试集R2:' + str(test_r2))
