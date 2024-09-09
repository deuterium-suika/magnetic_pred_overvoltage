# 对随机森林筛选的重要特征在不同的非正常状态参数条件下的变化曲线可视化

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


def plotVari1(dataPath1, dataPath2, dataPath3, dataPath4, dataPath5, dataPath6, ctype):
    # c相高压绕组感应电流在第14维，下标13
    data1 = np.loadtxt(dataPath1, encoding='utf-8', comments='%')[:, 13]
    data2 = np.loadtxt(dataPath2, encoding='utf-8', comments='%')[:, 13]
    data3 = np.loadtxt(dataPath3, encoding='utf-8', comments='%')[:, 13]
    data4 = np.loadtxt(dataPath4, encoding='utf-8', comments='%')[:, 13]
    data5 = np.loadtxt(dataPath5, encoding='utf-8', comments='%')[:, 13]
    data6 = np.loadtxt(dataPath6, encoding='utf-8', comments='%')[:, 13]

    time_data = np.linspace(1.16, 1.2, 81)   # (81,)

    # totalA_av = moving_average(totalA, 10)
    # totalB_av = moving_average(totalB, 10)
    # totalC_av = moving_average(totalC, 10)

    fig = plt.figure(figsize=(12, 12))  #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    pl.plot(time_data, data1, 'r', label=u'1.00 times')
    pl.plot(time_data, data2, 'gold', label=u'1.10 times')
    pl.plot(time_data, data3, 'darkgreen', label=u'1.20 times')
    pl.plot(time_data, data4, 'darkblue', label=u'1.30 times')
    pl.plot(time_data, data5, 'darkviolet', label=u'1.40 times')
    pl.plot(time_data, data6, 'slategrey', label=u'1.60 times')
    pl.legend()
    pl.legend(fontsize=16)
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width, box.width * 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
    pl.xlabel(u't/s')
    pl.ylabel(ctype + '/A')
    # plt.title(ctype + ' variation')
    plt.savefig('../figure/InputVari/' + ctype + '_variation.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()


def plotVari2(dataPath1, dataPath2, dataPath3, dataPath4, dataPath5, dataPath6, ctype):
    # b相高压绕组感应电压在第4维，下标3
    data1 = np.loadtxt(dataPath1, encoding='utf-8', comments='%')[:, 3]
    data2 = np.loadtxt(dataPath2, encoding='utf-8', comments='%')[:, 3]
    data3 = np.loadtxt(dataPath3, encoding='utf-8', comments='%')[:, 3]
    data4 = np.loadtxt(dataPath4, encoding='utf-8', comments='%')[:, 3]
    data5 = np.loadtxt(dataPath5, encoding='utf-8', comments='%')[:, 3]
    data6 = np.loadtxt(dataPath6, encoding='utf-8', comments='%')[:, 3]

    time_data = np.linspace(1.16, 1.2, 81)  # (81,)
    # totalA_av = moving_average(totalA, 10)
    # totalB_av = moving_average(totalB, 10)
    # totalC_av = moving_average(totalC, 10)

    fig = plt.figure(figsize=(12, 12))  # figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    pl.plot(time_data, data1, 'r', label=u'1.00 times')
    pl.plot(time_data, data2, 'gold', label=u'1.10 times')
    pl.plot(time_data, data3, 'darkgreen', label=u'1.20 times')
    pl.plot(time_data, data4, 'darkblue', label=u'1.30 times')
    pl.plot(time_data, data5, 'darkviolet', label=u'1.40 times')
    pl.plot(time_data, data6, 'slategrey', label=u'1.60 times')

    pl.legend()
    pl.legend(fontsize=16)
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width, box.width * 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
    pl.xlabel(u't/s')
    pl.ylabel(ctype + '/V')
    # plt.title(ctype + ' variation')
    plt.savefig('../figure/InputVari/' + ctype + '_variation.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # c相高压绕组感应电流和b相高压绕组感应电压重要度最高
    dataPath1 = '../data/1.0VacInput.txt'
    dataPath2 = '../data/1.1VacInput.txt'
    dataPath3 = '../data/1.2VacInput.txt'
    dataPath4 = '../data/1.3VacInput.txt'
    dataPath5 = '../data/1.4VacInput.txt'
    dataPath6 = '../data/1.6VacInput.txt'

    plotVari1(dataPath1, dataPath2, dataPath3, dataPath4, dataPath5, dataPath6,  'Current in the high voltage winding of C-phase')
    plotVari2(dataPath1, dataPath2, dataPath3, dataPath4, dataPath5, dataPath6,  'Induced voltage in the high voltage winding of B-phase')


