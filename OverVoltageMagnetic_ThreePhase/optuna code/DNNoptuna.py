# 使用optuna调参
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader   # 将data读取存储在loader中
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math
import datetime
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rc("font", family="Microsoft YaHei")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 32
EPOCHES = 10000
INNUMS = 16   # 输入维度
OUTNUMS = 350  # 输出维度

alpha1 = 1 / 4   # 限定epoch
alpha2 = 1 / 2
alpha3 = 3 / 4


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 3, 6)
    layers = []
    in_features = 16
    for i in range(n_layers):
        # out_features = trial.suggest_int("n_units_l{}".format(i), 16, 1024)
        out_features = trial.suggest_categorical("n_units_l{}".format(i), [32, 64, 128, 256, 512, 1024])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features

    layers.append(nn.Linear(in_features, OUTNUMS))
    # print(layers)
    return nn.Sequential(*layers)


def objective(trial):
    # model = define_model(trial).to(DEVICE)
    model = define_model(trial).to(DEVICE)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr1 = trial.suggest_float("lr1", 1e-6, lr, log=True)
    lr2 = trial.suggest_float("lr2", 1e-7, lr1, log=True)
    lr3 = trial.suggest_float("lr3", 1e-8, lr2, log=True)
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=lr1)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=lr2)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=lr3)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()   # 正常数据使用L1loss

    trX = np.loadtxt(trainXPath)
    trY = np.loadtxt(trainYPath)
    tr = zip(trX, trY)
    tr = list(tr)
    random.shuffle(tr)
    trX, trY = zip(*tr)
    trX = torch.Tensor(trX)
    trY = torch.Tensor(trY)
    trX = trX.to(DEVICE)
    trY = trY.to(DEVICE)
    torch_datasets = TensorDataset(trX, trY)
    loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_datasets,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=0,
    )

    testX = np.loadtxt(testXPath)
    testY = np.loadtxt(testYPath)
    testX = torch.Tensor(testX)
    testY = torch.Tensor(testY)
    testX = testX.to(DEVICE)
    testY = testY.to(DEVICE)
    # writer = SummaryWriter(logs)

    for epoch in range(EPOCHES):
        for step, (batchX, batchY) in enumerate(loader):
            if torch.cuda.is_available():
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            # print(batchX.shape)    # torch.Size([8, 1])
            y_pred = model(batchX)
            loss = loss_fn(y_pred, batchY)
        if (epoch + 1) % 10 == 0:
            print("Epoch", epoch + 1, loss.item())
            if epoch < alpha1 * EPOCHES:
                optimizer.zero_grad()
                loss.backward()  # backpass
                optimizer.step()  # gradient descent
            elif epoch < alpha2 * EPOCHES:
                optimizer1.zero_grad()
                loss.backward()  # backpass
                optimizer1.step()  # gradient descent
            elif epoch < alpha3 * EPOCHES:
                optimizer2.zero_grad()
                loss.backward()  # backpass
                optimizer2.step()  # gradient descent
            else:
                optimizer3.zero_grad()
                loss.backward()  # backpass
                optimizer3.step()  # gradient descent

        if torch.cuda.is_available():
            testX = testX.cuda()
        with torch.no_grad():
            predY = model(testX)
        loss = loss_fn(predY, testY)

        # 对无望的trail进行剪枝
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    # writer.close()

    if torch.cuda.is_available():
        testX = testX.cuda()
    with torch.no_grad():
        predY = model(testX)

    # 保存模型权重文件
    # torch.save(model, os.path.join(projectPath, "ModelPart{}".format(i), "F01_Model",
    #                              "Part{0}_DNN{1}.ptl".format(i, timeStr)))

    loss = loss_fn(predY, testY)
    # mape = mape_fn_tensor(testY, predY)
    predY = predY.data.cpu().numpy()

    header = "loss\t" + str(float(loss)) + "\n"
    for key, value in trial.params.items():
        header = header + "{}\t{}".format(key, value) + "\n"

    np.savetxt(projectPath + '/optuna result/DNN/trail{}.txt'.format(trial.number), predY, header=header)

    return loss

if __name__ == "__main__":
    # 这也没有交叉验证啊...还要自己手动设置I的值，自己事先划分完成然后通过修改I完成每一折的实验...
    I = 9
    projectPath = r'F:\Pycharm Projects\OverVoltageMagnetic_ThreePhase'
    trainXPath = projectPath + '/data/zstrainInput.txt'
    trainYPath = projectPath + '/data/trainPCA.txt'
    testXPath = projectPath + '/data/zstestInput.txt'
    testYPath = projectPath + '/data/testPCA.txt'

    """参数优化开始"""
    study = optuna.create_study(direction="minimize", study_name='net_trail{}'.format(I))
    studyName = study.study_name  # study_name未设置默认为None，这是干什么....

    study.optimize(objective, n_trials=20, timeout=60 * 60 * 200)  # 时间秒s
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
    print("  Value: ", trial.value)

    print("  Params: ")
    detail = "loss\t" + str(float(trial.value)) + "\n" \
             + "Epochs\t"+ str(float(EPOCHES)) + "\n" \
             + "batchSize\t" + str(float(BATCHSIZE)) + "\n" \
             # + "lr\t" + str(float(lr)) + "\n" \
             # + "lr1\t" + str(float(lr1)) + "\n" \
             # + "lr2\t" + str(float(lr2)) + "\n"\
             # + "lr3\t" + str(float(lr3)) + "\n"
    # detail = detail + "n_layers\t{}\n".format(len(nlayers))
    # for layeri in(range(len(nlayers))):
    #     detail = detail + "n_units_l{}\t{}\n".format(layeri, nlayers[layeri])
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        detail = detail + "{}\t{}".format(key, value) + "\n"

    os.makedirs(projectPath + '/optuna result/DNN/BestTrial{}'.format(trial.number))
    picSavePath = projectPath + '/optuna result//DNN/BestTrial{}'.format(trial.number)
    # predYData = np.loadtxt(projectPath + '/optuna_result/BestTrial{}.txt'.format(trial.number))

    # results = showPredY(predYData, I, picSavePath)

    with open(os.path.join(picSavePath, "BestTrial{}.txt".format(trial.number)),
              "a") as file:
        file.write("\n" + "\n-------------\n\n" + "\n" + detail + "\n-------------")


    '''
    """
    手动查看各模型表现
    """
    # 要查看的trial
    studyName = "no-name-ee2652fe-519b-4a77-b866-058b4db663b9"
    trial_number = 65
    detail = "Epochs\t"+ str(float(EPOCHES)) + "\n" \
             + "batchSize\t" + str(float(BATCHSIZE)) + "\n" \
             # + "lr\t" + str(float(lr)) + "\n" \
             # + "lr1\t" + str(float(lr1)) + "\n" \
             # + "lr2\t" + str(float(lr2)) + "\n"\
             # + "lr3\t" + str(float(lr3)) + "\n"
    detail = detail + "n_layers\t{}\n".format(len(nlayers))
    for i in(range(len(nlayers))):
        detail = detail + "n_units_l{}\t{}\n".format(i, nlayers[i])
    if not os.path.exists(pathRow + r"\..\F02_Results\{}\Trial{}".format(studyName, trial_number)):
        os.makedirs(pathRow + r"\..\F02_Results\{}\Trial{}".format(studyName, trial_number))
    picSavePath = pathRow + r"\..\F02_Results\{}\Trial{}".format(studyName, trial_number)
    predYData = np.loadtxt(pathRow + r"\..\F02_Results\{}\trial{}.txt".format(studyName, trial_number))
    # print(predYData.shape)
    results = showPredY(predYData, I, picSavePath)

    with open(os.path.join(picSavePath, "Trial{}.txt".format(trial_number)),
              "a") as file:
        file.write("\n" + results + "\n-------------\n\n" + "\n" + detail + "\n-------------")
    '''