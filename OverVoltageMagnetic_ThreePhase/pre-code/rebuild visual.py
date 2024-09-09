import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
import numpy as np
from matplotlib import rcParams

config = {
    "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 16,  # medium/large/small
    'font.style': 'normal',  # normal/italic/oblique
    'font.weight': 'bold',  # bold
    "mathtext.fontset": 'cm',  # 'cm' (Computer Modern)
    "font.serif": ['Times New Roman'],  # 'Simsun'宋体
    "axes.unicode_minus": False,  # 用来正常显示负号
         }
rcParams.update(config)


def plot_point(plot_data, ctype):
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = Axes3D(fig)
    x = plot_data[:, 0]
    y = plot_data[:, 1]
    z = plot_data[:, 2]
    color = plot_data[:, 3]
    max_color = np.max(color)
    colorband = max_color - np.min(color)

    jet = plt.cm.get_cmap('jet')
    sc = ax.scatter(x, y, z, vmin=np.min(color), vmax=max_color, s=1, c=color, cmap=jet)

    cbar = plt.colorbar(sc)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cbar.locator = tick_locator
    cbar.set_ticks([np.min(color), np.min(color) + 0.25 * colorband, np.min(color) + 0.5 * colorband,
                    np.min(color) + 0.75 * colorband, max_color])
    # cbar.formatter.set_scientific(True)  # 设置科学计数法
    cbar.formatter.set_powerlimits((0, 0))  # 设置colorbar为科学计数法
    cbar.set_label("B/T")
    cbar.update_ticks()
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.set_zlabel('z/mm')
    # plt.title(ctype + 's_real')
    # plt.tight_layout()
    plt.savefig('../figure/rebuild/' + ctype + 's_rebuild.jpg', bbox_inches='tight')  # 设置bbox_inches解决显示不全
    # plt.show()
    plt.close()

def plot_diff(plot_data, row_data, ctype):
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = Axes3D(fig)
    x = plot_data[:, 0]
    y = plot_data[:, 1]
    z = plot_data[:, 2]
    color = np.abs(row_data - plot_data[:, 3])
    max_color = np.max(color)
    colorband = max_color - np.min(color)

    jet = plt.cm.get_cmap('jet')
    sc = ax.scatter(x, y, z, vmin=np.min(color), vmax=max_color, s=1, c=color, cmap=jet)

    cbar = plt.colorbar(sc)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cbar.locator = tick_locator
    cbar.set_ticks([np.min(color), np.min(color) + 0.25 * colorband, np.min(color) + 0.5 * colorband,
                    np.min(color) + 0.75 * colorband, max_color])
    # cbar.formatter.set_scientific(True)  # 设置科学计数法
    cbar.formatter.set_powerlimits((0, 0))  # 设置colorbar为科学计数法
    cbar.set_label("B/T")
    cbar.update_ticks()
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.set_zlabel('z/mm')
    # plt.title(ctype + 's_real')
    # plt.tight_layout()
    plt.savefig('../figure/diff_rebuild/' + ctype + 's_rebuild_diff.jpg', bbox_inches='tight')  # 设置bbox_inches解决显示不全
    # plt.show()
    plt.close()


def pre_visual():
    # 时间为t=1.16到1.2，时间间隔为5e-4,共81条数据
    data_file = np.loadtxt('../pca result/rebuild_pca.txt')
    axis = np.loadtxt('../data/raw data/1.0/三相1.0.txt', encoding='utf-8', comments='%')[:, :3]
    # print(data_file.shape)   # (772, 1913)
    trainpath = '../data/train data/output'
    flielist = os.listdir(trainpath)
    i = 0
    for file in flielist:
        mag_data = data_file[i, :]
        mag_data = np.c_[axis, mag_data.T]
        raw_data = np.loadtxt(os.path.join(trainpath, file))[:, 3]
        filename, filetype = os.path.splitext(file)
        plot_point(mag_data, filename)
        plot_diff(mag_data, raw_data, filename)
        i += 1


if __name__ == '__main__':
    pre_visual()
