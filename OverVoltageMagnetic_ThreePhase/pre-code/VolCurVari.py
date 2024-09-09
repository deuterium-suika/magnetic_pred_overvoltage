# 对于输入的参数随时间变化的情况做可视化
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams

config = {
    "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 24,  # medium/large/small
    'font.style': 'normal',  # normal/italic/oblique
    'font.weight': 'bold',  # bold
    "mathtext.fontset": 'cm',  # 'cm' (Computer Modern)
    "font.serif": ['Times New Roman'],  # 'Simsun'宋体
    "axes.unicode_minus": False,  # 用来正常显示负号
         }
rcParams.update(config)


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def plotCurVari(dataPath, ctype):
    data = np.loadtxt(dataPath, encoding='utf-8', comments='%')
    time_data = np.linspace(1.16, 1.2, 81)   # (81,)
    totalA = np.zeros((1,1))
    totalB = np.zeros((1, 1))
    totalC = np.zeros((1, 1))
    for i in range(81):
        dataA = data[3 * i]
        dataB = data[3 * i + 1]
        dataC = data[3 * i + 2]

        totalA = np.c_[totalA, dataA]
        totalB = np.c_[totalB, dataB]
        totalC = np.c_[totalC, dataC]
    totalA = np.delete(totalA, 0, axis=1).T
    totalB = np.delete(totalB, 0, axis=1).T
    totalC = np.delete(totalC, 0, axis=1).T

    # totalA_av = moving_average(totalA, 10)
    # totalB_av = moving_average(totalB, 10)
    # totalC_av = moving_average(totalC, 10)

    fig = plt.figure(figsize = (12,12))    #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    pl.plot(time_data,totalA,'r-*',label=u'A')
    p2 = pl.plot(time_data, totalB,'g-^', label=u'B')
    p3 = pl.plot(time_data, totalC, 'b-.', label=u'C')
    pl.legend()
    pl.legend(fontsize=16)
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width, box.width * 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
    pl.xlabel(u't/s')
    pl.ylabel(ctype + '/A')
    plt.ylim(1.1 * min(totalA), 1.21 * max(totalA))
    # plt.title(ctype + ' variation')
    plt.savefig('../figure/InputVari/' + ctype + '_variation.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()

def plotVolVari(dataPath, ctype):
    data = np.loadtxt(dataPath, encoding='utf-8', comments='%')
    time_data = np.linspace(1.16, 1.2, 81)   # (81,)
    totalA = np.zeros((1,1))
    totalB = np.zeros((1, 1))
    totalC = np.zeros((1, 1))
    for i in range(81):
        dataA = data[3 * i]
        dataB = data[3 * i + 1]
        dataC = data[3 * i + 2]

        totalA = np.c_[totalA, dataA]
        totalB = np.c_[totalB, dataB]
        totalC = np.c_[totalC, dataC]
    totalA = np.delete(totalA, 0, axis=1).T
    totalB = np.delete(totalB, 0, axis=1).T
    totalC = np.delete(totalC, 0, axis=1).T

    # totalA_av = moving_average(totalA, 10)
    # totalB_av = moving_average(totalB, 10)
    # totalC_av = moving_average(totalC, 10)

    fig = plt.figure(figsize = (12,12))    #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    pl.plot(time_data,totalA,'r-*',label=u'A')
    p2 = pl.plot(time_data, totalB,'g-^', label=u'B')
    p3 = pl.plot(time_data, totalC, 'b-.', label=u'C')
    pl.legend()
    pl.legend(fontsize=16)
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width, box.width * 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
    pl.xlabel(u't/s')
    pl.ylabel(ctype + '/V')
    plt.ylim(1.1 * min(totalA), 1.2 * max(totalA))
    # plt.title(ctype + ' variation')
    plt.savefig('../figure/InputVari/' + ctype + '_variation.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    dataPath1 = '../data/raw data/1.0/H1.txt'
    dataPath2 = '../data/raw data/1.0/HV1.txt'
    dataPath3 = '../data/raw data/1.0/L1.txt'
    dataPath4 = '../data/raw data/1.0/LV1.txt'
    dataPath5 = '../data/raw data/1.0/M1.txt'
    dataPath6 = '../data/raw data/1.0/MV1.txt'
    plotCurVari(dataPath1, 'Current in the high voltage winding')

    plotVolVari(dataPath2, 'Induced voltage in the high voltage winding')
    # plotCurVari(dataPath3, '低压绕组感应电流')
    plotVolVari(dataPath4, 'Induced voltage in the low voltage winding')
    plotCurVari(dataPath5, 'Current in the intermediate voltage winding')
    plotVolVari(dataPath6, 'Induced voltage in the intermediate voltage winding')
