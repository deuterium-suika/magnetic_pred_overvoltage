import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
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



def rebuildPCA(lowdim_data):
    # pca_data = np.loadtxt('../result/mix/LR/predY_linearmodel.txt', encoding='utf-8', comments='#')
    mean_pca = np.loadtxt('../pca result/mean_pca.txt', encoding='utf-8', comments='%')
    vector_pca = np.loadtxt('../pca result/vector_pca.txt', encoding='utf-8', comments='%')
    print(lowdim_data.shape)    # (17, 64)
    print(vector_pca.shape)    # (64, 21912)
    print(mean_pca.shape)   # (21912,)
    huanyuan_data = np.matmul(lowdim_data, vector_pca) + mean_pca
    np.savetxt('../result/DNN/predY_DNNmodel_rebuild.txt', huanyuan_data)
    return huanyuan_data


# 需要还原后计算mape
def mape_fn(yTrue, yPred):
    # for i in range(yTrue)
    return np.mean(np.abs((yPred - yTrue) / yTrue)) * 100


def visualize_point(plot_data, ctype):
    # 预测结果可视化
    visual_axis = np.loadtxt('../data/raw data/1.0/三相1.0.txt', comments='%', encoding='utf-8')[:, :3]

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = Axes3D(fig)
    # plot_data = np.loadtxt(file_path, comments='%', encoding='utf-8')
    x = visual_axis[:, 0]
    y = visual_axis[:, 1]
    z = visual_axis[:, 2]
    color = plot_data
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
    # plt.title(ctype + 's_pred')
    # plt.tight_layout()
    plt.savefig('../figure/predict_DNN/' + ctype + 's_pred.jpg', bbox_inches='tight')  # 设置bbox_inches解决显示不全
    # plt.show()
    plt.close()


def visualize_diff(pred_data, test_data, ctype):
    visual_axis = np.loadtxt('../data/raw data/1.0/三相1.0.txt', comments='%', encoding='utf-8')[:, :3]
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = Axes3D(fig)
    # plot_data = np.loadtxt(file_path, comments='%', encoding='utf-8')
    x = visual_axis[:, 0]
    y = visual_axis[:, 1]
    z = visual_axis[:, 2]
    color = np.abs(test_data - pred_data)
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
    # plt.title(ctype + 's_diff')
    # plt.tight_layout()
    plt.savefig('../figure/diff_DNN/' + ctype + 's_diff.jpg', bbox_inches='tight')  # 设置bbox_inches解决显示不全
    # plt.show()
    plt.close()


def visual_pred():
    '''
    预测结果可视化，包括误差的可视化
    :return:
    '''
    filepath = '../data/test data/output'
    filelist = os.listdir(filepath)
    i = 0
    mape_list = []
    for file in filelist:
        filename, filetype = os.path.splitext(file)
        print(filename)
        pred_data = np.loadtxt('../result/DNN/predY_DNNmodel_rebuild.txt')[i, :].T
        test_data = np.loadtxt(os.path.join(filepath, file))[:, 3]
        filemape = mape_fn(pred_data, test_data)
        print(filemape)
        visualize_point(pred_data, filename)
        visualize_diff(pred_data, test_data, filename)
        mapelist = filename + ':  ' + str(filemape)
        mape_list.append(mapelist)
        i += 1
        print(mape_list)
    mape_list = np.array(mape_list)
    np.savetxt('./result/DNNportal/mapelist.txt', mape_list)

if __name__ == '__main__':
    y_pre = np.loadtxt('../result/DNNportal/predY_2023-03-27-21-40-49.txt', comments='#', delimiter=',')
    testOut = np.loadtxt('../data/testOutput.txt')
    rebuildy_pre = rebuildPCA(y_pre)

    test_mae = mean_absolute_error(testOut, rebuildy_pre)
    test_mape = mape_fn(testOut, rebuildy_pre)
    test_mse = mean_squared_error(testOut, rebuildy_pre)
    test_r2 = r2_score(testOut, rebuildy_pre)
    print('测试集RMSE: ', np.sqrt(mean_squared_error(testOut, rebuildy_pre)))
    print('测试集MSE:' + str(test_mse))
    print('测试集MAE:' + str(test_mae))
    print('测试集MAPE:' + str(test_mape))
    print('测试集R2:' + str(test_r2))

    visual_pred()
