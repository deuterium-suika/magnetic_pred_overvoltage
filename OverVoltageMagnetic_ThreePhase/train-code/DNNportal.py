# -*- coding: utf-8 -*-
# 发热功率密度数据PCA降维后保留的主成分在第一个维度上就保持在1e+8或者1e+7的数量级，第二个维度开始降到1e-2和1e-1数量级
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader   # 将data读取存储在loader中
from torch.utils.data import TensorDataset
import time

#训练集PCA结果
trainY_path = './data/trainPCA.txt'
#测试集标签（位置+电流）
testX_path = './data/zstestInput.txt'
testY_path = './data/testPCA.txt'
# test_input = test_input.reshape(-1, 1)
# 训练数据的标签（位置+电流）
trainX_path = './data/zstrainInput.txt'
predY_save_path = './result/DNN'


def get_parameter_number(net):    # 网络参数数量
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class NetShortCircuit(nn.Module):
    def __init__(self):
        super(NetShortCircuit, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 256)
        # self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(256, 128)
        # self.drop2 = nn.Dropout()
        self.fc4 = nn.Linear(128, 128)
        # self.drop3 = nn.Dropout()
        self.fc5 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 350)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.drop1(x)
        x = self.fc3(x)
        x = torch.relu(x)
        # x = self.drop2(x)
        x = self.fc4(x)
        x = torch.relu(x)
        # x = self.drop3(x)
        x = self.fc5(x)
        x = torch.relu(x)
        # x = self.fc6(x)
        # x = torch.relu(x)
        x = self.fc7(x)
        return x

lr = 0.00024269967765539285
lr1 = 0.0002098859263373215
lr2 = 0.00016243817742567272
lr3 = 9.665834424456342e-05
BATCH_SIZE = 80
epochs = 30000
net = NetShortCircuit()
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
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
optimizer1 = torch.optim.Adam(net.parameters(), lr = lr1)
optimizer2 = torch.optim.Adam(net.parameters(), lr = lr2)
optimizer3 = torch.optim.Adam(net.parameters(), lr = lr3)
# print(optimizer)

epoches = []
losses = []
testlosses = []

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

for epoch in range(epochs):
    # 每一个epoch内随机打乱数据，使每个batch内的数据尽量不重复
    for step, (batchX, batchY) in enumerate(train_loader):
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = net(batchX)
        loss = loss_fn(y_pred, batchY)

        if epoch < 0.25 * epochs:
            optimizer.zero_grad()
            loss.backward()  # backpass
            optimizer.step()  # gradient descent
        elif epoch < 0.5 * epochs:
            optimizer1.zero_grad()
            loss.backward()  # backpass
            optimizer1.step()
        elif epoch < 0.75 * epochs:
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

with torch.no_grad():
    predY = net(testX)

model_savePath = './model/DNN_{}.pth'.format(current_time)
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(), model_savePath)
# 针对上面一般的保存方法，加载的方法分别是：
# model_dict=model.load_state_dict(torch.load(PATH))

loss = loss_fn(predY, testY)   # 这里计算是否有问题？因为测试的MSE达到了几百
# mape = mape_fn(testY, predY)
detail ="time:"+str(current_time) +"\n"+ str(net) + "\n lr: " + str(lr) + \
        "\n optimizer:"+str(optimizer)+"\n BATCH_SIZE: "+str(BATCH_SIZE)+"\n epochs: " \
        + str(epochs) + "\n test_Loss:" + str(loss)
print(detail)
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
vari_loss = np.c_[np.c_[array_epoch, array_loss], array_testloss]
np.savetxt(os.path.join(predY_save_path,"LossVari_{}.txt".format(current_time)), vari_loss)

def mape_fn(yTrue, yPred):
    # for i in range(yTrue)
    return np.mean(np.abs((yPred - yTrue) / yTrue)) * 100

def rebuildPCA(lowdim_data):
    # pca_data = np.loadtxt('../result/mix/LR/predY_linearmodel.txt', encoding='utf-8', comments='#')
    mean_pca = np.loadtxt('./data/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('./data/vector_pca.txt', encoding='utf-8', comments='%')
    # print(lowdim_data.shape)
    # print(vector_pca.shape)
    huanyuan_data = np.matmul(lowdim_data, vector_pca) + mean_pca
    # np.savetxt('../result/mix/LR/predY_linearmodel_rebuild.txt', huanyuan_data)
    # print(huanyuan_data.shape)
    return huanyuan_data


testY = np.loadtxt('./data/testOutput.txt')

# rebuildtestY = rebuildPCA(testY)
rebuildy_pre = rebuildPCA(predY)

test_mape = mape_fn(testY, rebuildy_pre)
print('测试集MAPE:' + str(test_mape))