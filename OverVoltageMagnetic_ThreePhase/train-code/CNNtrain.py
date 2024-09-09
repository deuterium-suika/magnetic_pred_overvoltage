# -*- coding: utf-8 -*-
# 发热功率密度数据PCA降维后保留的主成分在第一个维度上就保持在1e+8或者1e+7的数量级，第二个维度开始降到1e-2和1e-1数量级
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader   # 将data读取存储在loader中
from torch.utils.data import TensorDataset
import time
import datetime

# 训练集PCA结果
trainY_path = '../data/trainPCA.txt'
# 测试集标签（位置+电流）
testX_path = '../data/zstestInput.txt'
testY_path = '../data/testPCA.txt'
# test_input = test_input.reshape(-1, 1)
# 训练数据的标签（位置+电流）
trainX_path = '../data/zstrainInput.txt'
# logs = "../log/DNN-org-pca"
predY_save_path = '../result/CNN'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def mape_fn(yTrue, yPred):
#     return torch.mean(torch.abs((yPred - yTrue) / yTrue)) * 100

#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class NetShortCircuit(nn.Module):
    def __init__(self):
        super(NetShortCircuit, self).__init__()
        # self.fc1 = nn.Linear(4, 8)   # (4, 1, 64)
        self.cov1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.cov2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.cov3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc2 = nn.Linear(320, 350)
        self.fc3 = nn.Linear(350, 350)

    def forward(self, x):
        # print(x)  # torch.Size([8, 1, 4])
        # x = F.relu(self.fc1(x))
        # print(x.size())   # torch.Size([8, 1, 16])
        # print(x)
        x = F.relu(self.cov1(x))
        # print(x.shape)   # torch.Size([8, 8, 14])
        x = self.max_pool1(x)
        # print(x.shape)   # ([8, 16, 12])
        x = F.relu(self.cov2(x))
        # print(x.shape)   # torch.Size([8, 32, 5])
        x = self.max_pool2(x)
        # print(x.shape)   # torch.Size([8, 32, 4])
        # x = F.relu(self.cov3(x))
        # print(x.shape)   # torch.Size([8, 32, 5])
        # x = self.max_pool3(x)
        # print(x.shape)   # torch.Size([8, 32, 5])
        x = x.view(-1, 32 * 10)
        # print(x.shape)
        # x = F.relu(self.fc6(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)
        return x



lr = 1e-4
BATCH_SIZE = 64
epochs = 10000
net = NetShortCircuit().to(device)
print(get_parameter_number(net))
# summary(net, input_size=(1, 1))
trX = np.loadtxt(trainX_path)
trY = np.loadtxt(trainY_path)
tr = zip(trX, trY)
tr = list(tr)
random.shuffle(tr)
trX, trY = zip(*tr)
trX = torch.Tensor(np.array(trX))
trY = torch.Tensor(np.array(trY))
# trX = trX.to(device)
# trY = trY.to(device)
# trX = trX.to(device)
# trY = trY.to(device)
torch_datasets = TensorDataset(trX, trY)
loader = DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_datasets,
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
#实例化，文件夹logs
# writer = SummaryWriter(logs)
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
start_time = datetime.datetime.now()
for epoch in range(epochs):
    # 每一个epoch内随机打乱数据，使每个batch内的数据尽量不重复

    for step, (batchX, batchY) in enumerate(loader):
        batchX = torch.unsqueeze(batchX, 1)
        # print(batchX.shape)   # torch.Size([8, 1, 4])
        # batchY = torch.unsqueeze(batchY, 0)
        # print(batchY.shape)    #
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = net(batchX)
        # print(y_pred.shape)
        # print(batchY.shape)
        loss = loss_fn(y_pred, batchY)
        # writer.add_scalar(current_time, loss.item(), epoch)
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
            testin = torch.unsqueeze(testin, 1)
            # print(batchX.shape)
            # testout = torch.unsqueeze(testout, 0)
            if torch.cuda.is_available():
                testin = testin.cuda()
                testout = testout.cuda()
            predY = net(testin)
            test_loss = loss_fn(predY, testout)  # 这里计算是否有问题？因为测试的MSE达到了几百
        if (epoch + 1) % 30 == 0:
            print("Epoch {} testingLoss".format(epoch + 1), test_loss.item())
    epoches.append(epoch + 1)
    losses.append(loss.item())

# writer.close()
end_time = datetime.datetime.now()
cost_time = str(end_time - start_time)

array_epoch = np.array(epoches)
array_loss = np.array(losses)
vari_loss = np.c_[array_epoch, array_loss]
np.savetxt(predY_save_path + '/LossVariationPCACPU-{}.txt'.format(current_time), vari_loss)

testX = np.loadtxt(testX_path)
testY = np.loadtxt(testY_path)
# print(testY.shape)
testX = torch.Tensor(np.array(testX))
testY = torch.Tensor(np.array(testY))
testX = torch.unsqueeze(testX, 1)
testX = testX.to(device)
testY = testY.to(device)
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():
    predY = net(testX)

model_savePath = '../modelCNN/CNNpcaCPU-{}.pth'.format(current_time)
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(), model_savePath)
# 针对上面一般的保存方法，加载的方法分别是：
# model_dict=model.load_state_dict(torch.load(PATH))


loss = loss_fn(predY, testY)   # 这里计算是否有问题？因为测试的MSE达到了几百
# mape = mape_fn(testY, predY)
detail = "time:" + str(current_time) + "\r\n" + str(net) + "\r\n lr: " + str(lr) + \
         "\r\n optimizer:" + str(optimizer) + "\r\n BATCH_SIZE: " + str(BATCH_SIZE) + "\r\n epochs: " \
         + str(epochs) + "\r\n total-trainable_num" + str(get_parameter_number(net)) + \
         "\r\n training_duration:" + cost_time + "\r\n test_Loss:" + str(loss) + "\r\n"
print(detail)
# predY = predY.data.cpu().numpy()

def mape_fn(yTrue, yPred):
    return np.mean(np.abs((yPred - yTrue) / yTrue)) * 100

def rebuildPCA(lowdim_data):
    mean_pca = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')

    huanyuan_data = np.matmul(lowdim_data, vector_pca) + mean_pca
    return huanyuan_data


predY = predY.data.cpu().numpy()


test_real = np.loadtxt('../data/testOutput.txt')
rebuildy_pre = rebuildPCA(predY)

test_mape = mape_fn(test_real, rebuildy_pre)
print('测试集MAPE:' + str(test_mape))

detail += "test MAPE:" + str(test_mape) + "\r\n"

np.savetxt(os.path.join(predY_save_path,"predYCNNpcaCPU-{}.txt".format(current_time)), predY,  delimiter=',', header=detail)
# np.savetxt(os.path.join(predY_save_path,"predY_{}_detail.txt".format(time)), 00, header=detail)
with open(os.path.join(predY_save_path,"ResultsCNNpcaCPU-{}.txt".format(current_time)),"a")as file:
    file.write("\r\n\r\n" + detail + "\r\n-----------------")
    file.close()