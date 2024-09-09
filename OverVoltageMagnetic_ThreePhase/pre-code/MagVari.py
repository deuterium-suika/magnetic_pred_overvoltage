# 所有点上的磁通均值随着时间参数变化的曲线
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/raw data/1.25/三相1.25.txt', comments='%', encoding='utf-8')
data = data[:, 3:]

time_data = np.linspace(0.36, 0.4, 81)
mean_data = np.zeros((1, 1))
for i in range(data.shape[1]):
    data_i = data[:, i]
    # print(np.mean(data_i))
    mean_data = np.r_[mean_data, np.mean(data_i).reshape(-1, 1)]
mean_data = np.delete(mean_data, 0, axis=0)
print(mean_data.shape)  # (4000, 1)
print(time_data.shape)  # (4000,)

# 第一个坐标点的磁通变化情况
first_data = data[0, :]
last_data = data[-1, :]

print(max(first_data))
print(min(first_data))

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('time/s')  # x轴标签
plt.ylabel('mag/T')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
# plt.plot(time_data, mean_data, linewidth=1, linestyle="solid", label="test loss")
plt.scatter(time_data, first_data, s=1)
# plt.legend()
plt.title('variation curve')
plt.show()


