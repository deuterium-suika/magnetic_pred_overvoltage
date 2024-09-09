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
predY_save_path = '../result/DNN'
logs = "../log"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameter_number(net):    # 网络参数数量
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class NetShortCircuit(nn.Module):
    def __init__(self):
        super(NetShortCircuit, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 256)
        # self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(256, 350)

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
BATCH_SIZE = 256
epochs = 5000
net = NetShortCircuit().to(device)
print(get_parameter_number(net))