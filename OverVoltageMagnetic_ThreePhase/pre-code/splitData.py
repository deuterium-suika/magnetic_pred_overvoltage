# 将电压和电流值分开，然后合并为18维特征
import numpy as np
import random
from sklearn import model_selection
from sklearn.decomposition import PCA
import os
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def splitFiles():
    '''
    将随时间变化混合到一起的磁场数据分开，分别保存到txt，输入也单独保存为txt
    :return:
    '''
    inputs = np.loadtxt('../data/allInput.txt')   # (405, 16)
    outputs = np.loadtxt('../data/allOutput.txt')    # (405, 21912)
    print(inputs.shape)
    axis = np.loadtxt('../data/raw data/1.0/三相1.0.txt', encoding='utf-8', comments='%')[:, :3]   # (21912, 3)
    t0 = 1.16
    for i in range(81):
        data1 = np.c_[axis, outputs[i, :].reshape(-1, 1)]   # 将axis和磁场混合到一起
        data2 = np.c_[axis, outputs[81 + i, :].reshape(-1, 1)]
        data3 = np.c_[axis, outputs[162 + i, :].reshape(-1, 1)]
        data4 = np.c_[axis, outputs[243 + i, :].reshape(-1, 1)]
        data5 = np.c_[axis, outputs[324 + i, :].reshape(-1, 1)]
        data6 = np.c_[axis, outputs[405 + i, :].reshape(-1, 1)]

        input1 = inputs[i, :]
        input2 = inputs[81 + i, :]
        input3 = inputs[162 + i, :]
        input4 = inputs[243 + i, :]
        input5 = inputs[324 + i, :]
        input6 = inputs[405 + i, :]
        # print(input1.shape)    # (16,)
        # print(data2.shape)   # (21912, 4)
        np.savetxt('../data/splited data/output/1.0-{:.4f}.txt'.format(t0 + 5e-4 * i), data1)
        np.savetxt('../data/splited data/input/1.0-{:.4f}.txt'.format(t0 + 5e-4 * i), input1)
        # np.savetxt('../data/splited data/input/1.0-{:.4f}.txt'.format(t0 + 5e-4 * i), input1, fmt='%10e %10e %10e %10e '
                                                            # '%10e %10e %10e %10e %10e %10e %10e %10e %10e %10e %10e %1e')

        np.savetxt('../data/splited data/output/1.1-{:.4f}.txt'.format(t0 + 5e-4 * i), data2)
        np.savetxt('../data/splited data/input/1.1-{:.4f}.txt'.format(t0 + 5e-4 * i), input2)

        np.savetxt('../data/splited data/output/1.2-{:.4f}.txt'.format(t0 + 5e-4 * i), data3)
        np.savetxt('../data/splited data/input/1.2-{:.4f}.txt'.format(t0 + 5e-4 * i), input3)

        np.savetxt('../data/splited data/output/1.3-{:.4f}.txt'.format(t0 + 5e-4 * i), data4)
        np.savetxt('../data/splited data/input/1.3-{:.4f}.txt'.format(t0 + 5e-4 * i), input4)

        np.savetxt('../data/splited data/output/1.4-{:.4f}.txt'.format(t0 + 5e-4 * i), data5)
        np.savetxt('../data/splited data/input/1.4-{:.4f}.txt'.format(t0 + 5e-4 * i), input5)

        np.savetxt('../data/splited data/output/1.6-{:.4f}.txt'.format(t0 + 5e-4 * i), data6)
        np.savetxt('../data/splited data/input/1.6-{:.4f}.txt'.format(t0 + 5e-4 * i), input6)


def splitTrainTest():
    fileDir1 = '../data/splited data/output'
    data_list = os.listdir(fileDir1)  # 数据原始路径
    fileDir2 = '../data/splited data/input'
    data_number = len(data_list)
    train_number = int(data_number * 0.8)
    train_sample = random.sample(data_list, train_number)  # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(data_list) - set(train_sample))
    sample = [train_sample, test_sample]
    trainin_path = '../data/train data/input'
    testinpath = '../data/test data/input'
    trainoutpath = '../data/train data/output'
    testoutpath = '../data/test data/output'
    save_dir1 = [trainoutpath, testoutpath]
    save_dir2 = [trainin_path, testinpath]
    for k in range(len(save_dir1)):  # savedir是一个list，分别存储train和test的目标路径，第一次是复制train路径，然后复制test路径
        if os.path.isdir(save_dir1[k]):
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir1, name), os.path.join(save_dir1[k] + '/', name))
        else:
            os.makedirs(save_dir1[k])
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir1, name), os.path.join(save_dir1[k] + '/', name))

    for j in range(len(save_dir2)):
        if os.path.isdir(save_dir2[j]):
            for name in sample[j]:
                shutil.copy(os.path.join(fileDir2, name), os.path.join(save_dir2[j] + '/', name))
        else:
            os.makedirs(save_dir2[j])
            for name in sample[j]:
                shutil.copy(os.path.join(fileDir2, name), os.path.join(save_dir2[j] + '/', name))



def combineTrainTest():
    traininpath = '../data/train data/input'
    testinpath = '../data/test data/input'
    trainoutpath = '../data/train data/output'
    testoutpath = '../data/test data/output'

    filelist1 = os.listdir(traininpath)
    filelist2 = os.listdir(testinpath)
    filelist3 = os.listdir(trainoutpath)
    filelist4 = os.listdir(testoutpath)

    # 合并训练集的输入
    X_train = np.zeros((1, 16))
    y_train = np.zeros((1, 21912))   # 新的base坐标网格点数为3638
    X_test = np.zeros((1, 16))
    y_test = np.zeros((1, 21912))    # 新的base坐标网格点数为3638
    for file in filelist1:
        # print(file)
        datain = np.loadtxt(os.path.join(traininpath, file))
        X_train = np.r_[X_train, datain.reshape(1, -1)]
    for file in filelist2:
        datain = np.loadtxt(os.path.join(testinpath, file))
        X_test = np.r_[X_test, datain.reshape(1, -1)]
    for file in filelist3:
        dataout = np.loadtxt(os.path.join(trainoutpath, file))[:, 3]
        y_train = np.r_[y_train, dataout.reshape(1, -1)]
    for file in filelist4:
        dataout = np.loadtxt(os.path.join(testoutpath, file))[:, 3]
        y_test = np.r_[y_test, dataout.reshape(1, -1)]

    X_train = np.delete(X_train, 0, axis = 0)
    y_train = np.delete(y_train, 0, axis = 0)
    X_test = np.delete(X_test, 0, axis = 0)
    y_test = np.delete(y_test, 0, axis = 0)

    print(X_train.shape)   # (388, 16)
    print(y_train.shape)    # (388, 21912)
    print(X_test.shape)    # (98, 16)
    print(y_test.shape)   # (98, 21912)

    np.savetxt('../data/trainInput.txt', X_train)
    np.savetxt('../data/trainOutput.txt', y_train)
    np.savetxt('../data/testInput.txt', X_test)
    np.savetxt('../data/testOutput.txt', y_test)


def zscoreNormalize(data, mu, sigma):
    '''
    使用零均值（zero-score）归一化，把PCA降维后的所有维度数据映射到1个范围内，消除各维度特征之间数量级带来的训练权重偏向。
    但是这一步骤是否必须不确定，因为数量级较小的维度经过PCA还原后对原始数据的影响（因为对方差的贡献度问题）并不像PCA还原前那么明显（可能在后面的维度预测不准确，但是还原后的数据差异很小）
    :return:
    '''
    data = (data - mu) / sigma
    return data


def outdataNormalize():
    '''
    分别对训练和测试数据在1241个维度上进行降维，注意测试数据降维时使用的是训练数据的均值和标准差
    :return:
    '''
    trainPCAdata = np.loadtxt('../data/trainOutput.txt')   # (96, 1241)
    testPCAdata = np.loadtxt('../data/testOutput.txt')  # (24, 1241)
    # print(trainPCAdata.shape)

    # 对训练集输出做归一化
    new_trainPCAdata = np.zeros((64, 1))
    train_mu = np.zeros((1, 1))
    train_sigma = np.zeros((1, 1))
    for i in range(trainPCAdata.shape[1]):
        featurei = trainPCAdata[:, i]
        mui = np.mean(featurei).reshape(-1, 1)
        sigmai = np.std(featurei).reshape(-1, 1)
        zsi = zscoreNormalize(featurei, mui, sigmai).reshape(64, 1)
        new_trainPCAdata = np.c_[new_trainPCAdata, zsi]
        # print(new_trainPCAdata.shape)
        train_mu = np.c_[train_mu, mui]
        train_sigma = np.c_[train_sigma, sigmai]

    new_trainPCAdata = np.delete(new_trainPCAdata, 0, axis=1)
    train_mu = np.delete(train_mu, 0, axis=1)
    train_sigma = np.delete(train_sigma, 0, axis=1)
    print(new_trainPCAdata.shape)
    print(train_mu.shape)
    print(train_sigma.shape)

    np.savetxt('../data/zstrainPCA.txt', new_trainPCAdata)
    np.savetxt('../data/zstrainmu.txt', train_mu)
    np.savetxt('../data/zstrainsigma.txt', train_sigma)

    # 测试集的归一化有问题
    # 测试集数据归一化（利用训练集数据的mu和sigma
    new_testPCAdata = np.zeros((17, 1))
    for j in range(testPCAdata.shape[1]):
        featurej = testPCAdata[:, j]
        muj = train_mu[:, j]
        sigmaj = train_sigma[:, j]
        zsj = zscoreNormalize(featurej, muj, sigmaj).reshape(17, 1)
        print(zsj.shape)
        print(max(zsj))
        new_testPCAdata = np.c_[new_testPCAdata, zsj]
    new_testPCAdata = np.delete(new_testPCAdata, 0 ,axis=1)
    print(new_testPCAdata.shape)
    np.savetxt('../data/zstestPCA.txt', new_testPCAdata)


def indataNormalize():
    '''
    分别对训练和测试数据在15个输入维度（去除空载的低压绕组电流）上进行降维，注意测试数据降维时使用的是训练数据的均值和标准差
    过电压参数不需要归一化
    :return:
    '''
    trainInputdata = np.loadtxt('../data/trainInput.txt')   # (324, 16)
    testInputdata = np.loadtxt('../data/testInput.txt')  # (81, 16)

    # 对训练集做归一化
    new_trainindata = np.zeros((388, 1))
    train_mu = np.zeros((1, 1))
    train_sigma = np.zeros((1, 1))
    for i in range(trainInputdata.shape[1] - 1):   # 最后一维不做
        featurei = trainInputdata[:, i]  # 在4个输入特征维度上分别做归一化
        mui = np.mean(featurei).reshape(-1, 1)
        sigmai = np.std(featurei).reshape(-1, 1)
        zsi = zscoreNormalize(featurei, mui, sigmai).reshape(-1, 1)
        new_trainindata = np.c_[new_trainindata, zsi]
        # print(new_trainPCAdata.shape)
        train_mu = np.c_[train_mu, mui]
        train_sigma = np.c_[train_sigma, sigmai]

    new_trainindata = np.delete(new_trainindata, 0, axis=1)  # (324, 15)
    new_trainindata = np.c_[new_trainindata, trainInputdata[:, -1]]
    print(new_trainindata.shape)    #
    train_mu = np.delete(train_mu, 0, axis=1)   # (1, 15)
    train_sigma = np.delete(train_sigma, 0, axis=1)   # (1, 15)

    np.savetxt('../data/zstrainInput.txt', new_trainindata)
    np.savetxt('../data/zstrainmuInput.txt', train_mu)
    np.savetxt('../data/zstrainsigmaInput.txt', train_sigma)

    # 测试集的归一化有问题
    # 测试集数据归一化（利用训练集数据的mu和sigma
    new_testdata = np.zeros((98, 1))
    for j in range(testInputdata.shape[1] - 1):
        featurej = testInputdata[:, j]
        muj = train_mu[:, j]
        sigmaj = train_sigma[:, j]
        zsj = zscoreNormalize(featurej, muj, sigmaj).reshape(-1, 1)
        new_testdata = np.c_[new_testdata, zsj]
    new_testdata = np.delete(new_testdata, 0 ,axis=1)
    new_testdata = np.c_[new_testdata, testInputdata[:, -1]]
    print(new_testdata.shape)
    np.savetxt('../data/zstestInput.txt', new_testdata)



def trainPCA(trainPath):
    '''
    训练PCA
    :param trainPath:
    :return:
    '''
    value_c = np.loadtxt(trainPath)  # 3200 * 1241
    # print(value_c.shape)
    pca = PCA(n_components=350)
    new_data = pca.fit_transform(value_c)
    # print(new_data.shape)
    # 路径需要修改
    np.savetxt('../data/trainPCA.txt', new_data)
    # PCA均值数据
    np.savetxt('../pca result/mean_pca.txt', pca.mean_)
    # PCA特征向量
    np.savetxt('../pca result/vector_pca.txt', pca.components_)
    # PCA成分占比
    np.savetxt('../pca result/variance_pca.txt', pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum(axis=0))
    # print('training data PCA finished!')


def testPCA(testPath):
    '''
    测试PCA
    :param testPath:
    :return:
    '''
    mean = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    components = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')
    test_value_c = np.loadtxt(testPath) # 21 * 1241
    test_data = test_value_c - mean.reshape(1, -1)
    pca_test = np.matmul(test_data, components.T)
    # print(pca_test.shape)
    np.savetxt('../data/testPCA.txt', pca_test)
    # print('testing data PCA finished!')


def mape_fn(yTrue, yRebuild):
    return np.mean(np.abs((yRebuild - yTrue) / yTrue)) * 100


def rebuild_PCA():
    pca_data = np.loadtxt('../data/trainPCA.txt', encoding='utf-8', comments='%')
    mean_pca = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')

    # print(vector_pca.shape)  # (15, 21912)
    # print(pca_data.shape)  # (64, 15)
    huanyuan_data = np.matmul(pca_data, vector_pca) + mean_pca

    raw_data = np.loadtxt('../data/trainOutput.txt') # (80,1241)

    # print(raw_data.shape)
    # print(huanyuan_data.shape)
    print('rmse: ' + str(np.sqrt(mean_squared_error(raw_data, huanyuan_data))))
    print('mae: ' + str(mean_absolute_error(raw_data, huanyuan_data)))
    print('mse: ' + str(mean_squared_error(raw_data, huanyuan_data)))
    print('mape: ' + str(mape_fn(raw_data, huanyuan_data)) + '%')
    print('r2: ' + str(r2_score(raw_data, huanyuan_data)))
    np.savetxt('../pca result/rebuild_pca.txt', huanyuan_data)


# def test_rebuild():
#
#



if __name__ == '__main__':
    trainPath = '../data/trainOutput.txt'
    testPath = '../data/testOutput.txt'
    # combineInput()
    # removeInput()
    # getOutput()
    # splitFiles()
    # splitTrainTest()
    combineTrainTest()
    # outdataNormalize()
    indataNormalize()
    trainPCA(trainPath)
    testPCA(testPath)
    rebuild_PCA()

