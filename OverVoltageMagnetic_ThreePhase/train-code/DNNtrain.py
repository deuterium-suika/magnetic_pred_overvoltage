# -*- coding: utf-8 -*-
# 发热功率密度数据PCA降维后保留的主成分在第一个维度上就保持在1e+8或者1e+7的数量级，第二个维度开始降到1e-2和1e-1数量级
import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import random
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader   # 将data读取存储在loader中
from torch.utils.data import TensorDataset
import time
import math
import matplotlib.pyplot as plt
import pylab as pl

#训练集PCA结果
trainY_path = '../data/trainPCA.txt'
#测试集标签（位置+电流）
testX_path = '../data/zstestInput.txt'
testY_path = '../data/testPCA.txt'
# test_input = test_input.reshape(-1, 1)
# 训练数据的标签（位置+电流）
trainX_path = '../data/zstrainInput.txt'
logs = "../log"
predY_save_path = '../result/DNN'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameter_number(net):    # 网络参数数量
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class NetShortCircuit(nn.Module):
    def __init__(self):
        super(NetShortCircuit, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 1024)
        self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(512, 350)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        # x = self.fc6(x)
        # x = torch.relu(x)
        x = self.fc7(x)
        return x


lr = 1e-4
BATCH_SIZE = 64
epochs = 10000
net = NetShortCircuit().to(device)
trX = np.loadtxt(trainX_path)
trY = np.loadtxt(trainY_path)
trX = torch.Tensor(trX)
trY = torch.Tensor(trY)
train_datasets = TensorDataset(trX, trY)
train_loader = DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_datasets,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
testX = np.loadtxt(testX_path)
testY = np.loadtxt(testY_path)
testX = torch.Tensor(testX)
testY = torch.Tensor(testY)
test_datasets = TensorDataset(testX, testY)
test_loader = DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_datasets,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
 )
# 因为本身loss就下降的很快，加上PCA后的结果数量级也刚好在e-6，所以MAE会很快变小，但实际上可能并没有训练到位
# 这样也会导致MAEloss一直在e-6变化，那其实是网络学习能力不足了或是lr设置过大了
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
optimizer1 = torch.optim.Adam(net.parameters(), lr = 0.1 * lr)
optimizer2 = torch.optim.Adam(net.parameters(), lr = 0.01 * lr)
optimizer3 = torch.optim.Adam(net.parameters(), lr = 0.001 * lr)
# print(optimizer)

epoches = []
losses = []
testlosses = []
#实例化，文件夹logs
writer = SummaryWriter(logs)
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

for epoch in range(epochs):
    # 每一个epoch内随机打乱数据，使每个batch内的数据尽量不重复
    for step, (batchX, batchY) in enumerate(train_loader):
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = net(batchX)
        loss = loss_fn(y_pred, batchY)

        if epoch < 0.2 * epochs:
            optimizer.zero_grad()
            loss.backward()  # backpass
            optimizer.step()  # gradient descent
        elif epoch < 0.5 * epochs:
            optimizer1.zero_grad()
            loss.backward()  # backpass
            optimizer1.step()
        elif epoch < 0.7 * epochs:
            optimizer2.zero_grad()
            loss.backward()  # backpass
            optimizer2.step()
        else:
            optimizer3.zero_grad()
            loss.backward()  # backpass
            optimizer3.step()  # gradient descent
    if (epoch + 1) % 30 == 0:
        print("Epoch {} trainingLoss".format(epoch + 1), loss.item())
    with torch.no_grad():
        for testdata in test_loader:
            testin, testout = testdata
            if torch.cuda.is_available():
                testin = testin.cuda()
                testout = testout.cuda()
            predY = net(testin)
            test_loss = loss_fn(predY, testout)  # 这里计算是否有问题？因为测试的MSE达到了几百
        if (epoch + 1) % 30 == 0:
            print("Epoch {} testingLoss".format(epoch + 1), test_loss.item())
    epoches.append(epoch + 1)
    losses.append(loss.item())
    testlosses.append(test_loss.item())
writer.close()


if torch.cuda.is_available():
    testX = testX.cuda()
    testY = testY.cuda()
with torch.no_grad():
    predY = net(testX)

model_savePath = '../model/DNN_{}.pth'.format(current_time)
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(), model_savePath)
# 针对上面一般的保存方法，加载的方法分别是：
# model_dict=model.load_state_dict(torch.load(PATH))

loss = loss_fn(predY, testY)   # 这里计算是否有问题？因为测试的MSE达到了几百
# mape = mape_fn(testY, predY)
detail ="time:"+str(current_time) +"\n"+ str(net) + "\n lr: " + str(lr) + \
        "\n optimizer:"+str(optimizer)+"\n BATCH_SIZE: "+str(BATCH_SIZE)+"\n epochs: " \
        + str(epochs) + "\n test_Loss:" + str(loss)
# print(detail)
predY = predY.data.cpu().numpy()

#保存预测的特征值，用以还原原始磁场数据

np.savetxt(os.path.join(predY_save_path,"predY_{}.txt".format(current_time)), predY,  delimiter=',', header=detail)
# np.savetxt(os.path.join(predY_save_path,"predY_{}_detail.txt".format(time)), 00, header=detail)
with open(os.path.join(predY_save_path,"results_{}.txt".format(current_time)),"a")as file:
    file.write("\n\n"+detail+"\n-----------------")
    file.close()

array_epoch = np.array(epoches)
array_loss = np.array(losses)
array_testloss = np.array(testlosses)


fig = plt.figure(figsize = (7,5))    #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
pl.plot(array_epoch,array_loss,'g-',label=u'train loss')
p2 = pl.plot(array_epoch, array_testloss,'r-', label = u'test loss')
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'loss')
plt.title('Compare loss for different models in training')
plt.show()
