# 将数据按照过电压的大小从低到高依次合并到一起
# 包括数据参数16维和输出参数磁通的合并
import numpy as np

def readInput(h_vol, h_current, m_vol, m_current, l_vol):
    '''
    将ABC三相的电压与电流分开
    :return:
    '''
    # 1为h,2为m,3为l
    a1current = np.zeros((1,1))
    a2current = np.zeros((1,1))
    # a3current = np.zeros((1,1))
    a1vol = np.zeros((1,1))
    a2vol = np.zeros((1,1))
    a3vol = np.zeros((1,1))
    b1current = np.zeros((1,1))
    b2current = np.zeros((1,1))
    # b3current = np.zeros((1,1))
    b1vol = np.zeros((1,1))
    b2vol = np.zeros((1,1))
    b3vol = np.zeros((1,1))
    c1current = np.zeros((1,1))
    c2current = np.zeros((1,1))
    # c3current = np.zeros((1,1))
    c1vol = np.zeros((1,1))
    c2vol = np.zeros((1,1))
    c3vol = np.zeros((1,1))

    for i in range(81):
        a_h_current = h_current[3 * i]
        b_h_current = h_current[3 * i + 1]
        c_h_current = h_current[3 * i + 2]
        a_h_vol = h_vol[3 * i]
        b_h_vol = h_vol[3 * i + 1]
        c_h_vol = h_vol[3 * i + 2]

        a1current = np.c_[a1current, a_h_current]
        b1current = np.c_[b1current, b_h_current]
        c1current = np.c_[c1current, c_h_current]
        a1vol = np.c_[a1vol, a_h_vol]
        b1vol = np.c_[b1vol, b_h_vol]
        c1vol = np.c_[c1vol, c_h_vol]

        a_m_current = m_current[3 * i]
        b_m_current = m_current[3 * i + 1]
        c_m_current = m_current[3 * i + 2]
        a_m_vol = m_vol[3 * i]
        b_m_vol = m_vol[3 * i + 1]
        c_m_vol = m_vol[3 * i + 2]

        a2current = np.c_[a2current, a_m_current]
        b2current = np.c_[b2current, b_m_current]
        c2current = np.c_[c2current, c_m_current]
        a2vol = np.c_[a2vol, a_m_vol]
        b2vol = np.c_[b2vol, b_m_vol]
        c2vol = np.c_[c2vol, c_m_vol]

        a_l_vol = l_vol[3 * i]
        b_l_vol = l_vol[3 * i + 1]
        c_l_vol = l_vol[3 * i + 2]

        a3vol = np.c_[a3vol, a_l_vol]
        b3vol = np.c_[b3vol, b_l_vol]
        c3vol = np.c_[c3vol, c_l_vol]

    a1current = np.delete(a1current, 0, axis=1)
    a2current = np.delete(a2current, 0, axis=1)
    # a3current = np.delete(a3current, 0, axis=1)
    a1vol = np.delete(a1vol, 0, axis=1)
    a2vol = np.delete(a2vol, 0, axis=1)
    a3vol = np.delete(a3vol, 0, axis=1)
    b1current = np.delete(b1current, 0, axis=1)
    b2current = np.delete(b2current, 0, axis=1)
    # b3current = np.delete(b3current, 0, axis=1)
    b1vol = np.delete(b1vol, 0, axis=1)
    b2vol = np.delete(b2vol, 0, axis=1)
    b3vol = np.delete(b3vol, 0, axis=1)
    c1current = np.delete(c1current, 0, axis=1)
    c2current = np.delete(c2current, 0, axis=1)
    # c3current = np.delete(c3current, 0, axis=1)
    c1vol = np.delete(c1vol, 0, axis=1)
    c2vol = np.delete(c2vol, 0, axis=1)
    c3vol = np.delete(c3vol, 0, axis=1)
    # 拼接到一个array
    all_input = np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[np.r_[a1vol,
                a2vol], a3vol], b1vol], b2vol], b3vol], c1vol], c2vol], c3vol], a1current], a2current],
                b1current], b2current], c1current], c2current].T   # (81, 15)

    return all_input


def cutOUtIn():
    '''
    截取输出的磁场数据和输入数据（0.02-0.06）
    :return:
    '''
    # 1.0
    h_current1 = np.loadtxt('../data/raw data/1.0/H1.txt', comments='%', encoding='utf-8')
    h_vol1 = np.loadtxt('../data/raw data/1.0/HV1.txt', comments='%', encoding='utf-8')
    m_current1 = np.loadtxt('../data/raw data/1.0/M1.txt', comments='%', encoding='utf-8')
    m_vol1 = np.loadtxt('../data/raw data/1.0/MV1.txt', comments='%', encoding='utf-8')
    l_vol1 = np.loadtxt('../data/raw data/1.0/LV1.txt', comments='%', encoding='utf-8')
    # 1.1
    h_current2 = np.loadtxt('../data/raw data/1.1/HI1.1.txt', comments='%', encoding='utf-8')
    h_vol2 = np.loadtxt('../data/raw data/1.1/HV1.1.txt', comments='%', encoding='utf-8')
    m_current2 = np.loadtxt('../data/raw data/1.1/MI1.1.txt', comments='%', encoding='utf-8')
    m_vol2 = np.loadtxt('../data/raw data/1.1/MV1.1.txt', comments='%', encoding='utf-8')
    l_vol2 = np.loadtxt('../data/raw data/1.1/LV1.1.txt', comments='%', encoding='utf-8')
    # 1.2
    h_current3 = np.loadtxt('../data/raw data/1.2/HI1.2.txt', comments='%', encoding='utf-8')
    h_vol3 = np.loadtxt('../data/raw data/1.2/HV1.2.txt', comments='%', encoding='utf-8')
    m_current3 = np.loadtxt('../data/raw data/1.2/MI1.2.txt', comments='%', encoding='utf-8')
    m_vol3 = np.loadtxt('../data/raw data/1.2/MV1.2.txt', comments='%', encoding='utf-8')
    l_vol3 = np.loadtxt('../data/raw data/1.2/LV1.2.txt', comments='%', encoding='utf-8')
    # 1.3
    h_current4 = np.loadtxt('../data/raw data/1.3/HI1.3.txt', comments='%', encoding='utf-8')
    h_vol4 = np.loadtxt('../data/raw data/1.3/HV1.3.txt', comments='%', encoding='utf-8')
    m_current4 = np.loadtxt('../data/raw data/1.3/MI1.3.txt', comments='%', encoding='utf-8')
    m_vol4 = np.loadtxt('../data/raw data/1.3/MV1.3.txt', comments='%', encoding='utf-8')
    l_vol4 = np.loadtxt('../data/raw data/1.3/LV1.3.txt', comments='%', encoding='utf-8')
    # 1.4
    h_current5 = np.loadtxt('../data/raw data/1.4/HI1.4.txt', comments='%', encoding='utf-8')
    h_vol5 = np.loadtxt('../data/raw data/1.4/HV1.4.txt', comments='%', encoding='utf-8')
    m_current5 = np.loadtxt('../data/raw data/1.4/MI1.4.txt', comments='%', encoding='utf-8')
    m_vol5 = np.loadtxt('../data/raw data/1.4/MV1.4.txt', comments='%', encoding='utf-8')
    l_vol5 = np.loadtxt('../data/raw data/1.4/LV1.4.txt', comments='%', encoding='utf-8')
    # 1.6
    h_current6 = np.loadtxt('../data/raw data/1.6/HI1.6.txt', comments='%', encoding='utf-8')
    h_vol6 = np.loadtxt('../data/raw data/1.6/HV1.6.txt', comments='%', encoding='utf-8')
    m_current6 = np.loadtxt('../data/raw data/1.6/MI1.6.txt', comments='%', encoding='utf-8')
    m_vol6 = np.loadtxt('../data/raw data/1.6/MV1.6.txt', comments='%', encoding='utf-8')
    l_vol6 = np.loadtxt('../data/raw data/1.6/LV1.6.txt', comments='%', encoding='utf-8')

    # 组合的格式是高压电压、中压电压、低压电压、高压电流、中压电流 15维输入（低压电流为0已删除）
    allinput1 = readInput(h_vol1, h_current1, m_vol1, m_current1, l_vol1)
    allinput2 = readInput(h_vol2, h_current2, m_vol2, m_current2, l_vol2)
    allinput3 = readInput(h_vol3, h_current3, m_vol3, m_current3, l_vol3)
    allinput4 = readInput(h_vol4, h_current4, m_vol4, m_current4, l_vol4)
    allinput5 = readInput(h_vol5, h_current5, m_vol5, m_current5, l_vol5)
    allinput6 = readInput(h_vol6, h_current6, m_vol6, m_current6, l_vol6)

    # print(allinput1.shape)   # (81, 15)
    # 过电压系数
    r1 = np.array([1.0 for _ in range(81)]).reshape(-1, 1)
    r2 = np.array([1.1 for _ in range(81)]).reshape(-1, 1)
    r3 = np.array([1.2 for _ in range(81)]).reshape(-1, 1)
    r4 = np.array([1.3 for _ in range(81)]).reshape(-1, 1)
    r5 = np.array([1.4 for _ in range(81)]).reshape(-1, 1)
    r6 = np.array([1.6 for _ in range(81)]).reshape(-1, 1)
    # 直接用电源电压做标签
    # r1 = np.array([63508.52961085883 for _ in range(81)]).reshape(-1, 1)
    # r2 = np.array([69859.38257194472 for _ in range(81)]).reshape(-1, 1)
    # r3 = np.array([76210.2355330306 for _ in range(81)]).reshape(-1, 1)
    # r4 = np.array([82561.08849411648 for _ in range(81)]).reshape(-1, 1)
    # r5 = np.array([88911.94145520237 for _ in range(81)]).reshape(-1, 1)
    # r6 = np.array([101613.6473773741 for _ in range(81)]).reshape(-1, 1)

    indata1 = np.c_[allinput1, r1]
    indata2 = np.c_[allinput2, r2]
    indata3 = np.c_[allinput3, r3]
    indata4 = np.c_[allinput4, r4]
    indata5 = np.c_[allinput5, r5]
    indata6 = np.c_[allinput6, r6]

    # print(indata1.shape)   # (81, 16)
    # 保存各过电压条件下的输入数据（含过电压的标签 ）
    np.savetxt('../data/1.0VacInput.txt', indata1)
    np.savetxt('../data/1.1VacInput.txt', indata2)
    np.savetxt('../data/1.2VacInput.txt', indata3)
    np.savetxt('../data/1.3VacInput.txt', indata4)
    np.savetxt('../data/1.4VacInput.txt', indata5)
    np.savetxt('../data/1.6VacInput.txt', indata6)

    # 1.3 1.4排在1.6之前
    inputdata = np.r_[indata1, np.r_[indata2, np.r_[indata3, np.r_[indata4, np.r_[indata5, indata6]]]]]
    print(inputdata.shape)   # (405, 16)
    outdata1 = np.loadtxt('../data/raw data/1.0/三相1.0.txt', comments='%', encoding='utf-8')[:, 3:].T
    outdata2 = np.loadtxt('../data/raw data/1.1/三相1.1.txt', comments='%', encoding='utf-8')[:, 3:].T
    outdata3 = np.loadtxt('../data/raw data/1.2/三相1.2.txt', comments='%', encoding='utf-8')[:, 3:].T
    outdata4 = np.loadtxt('../data/raw data/1.3/三相1.3.txt', comments='%', encoding='utf-8')[:, 3:].T
    outdata5 = np.loadtxt('../data/raw data/1.4/三相1.4.txt', comments='%', encoding='utf-8')[:, 3:].T
    outdata6 = np.loadtxt('../data/raw data/1.6/三相1.6.txt', comments='%', encoding='utf-8')[:, 3:].T
    print(outdata2.shape)   # (21912, 81)
    outputdata = np.r_[outdata1, np.r_[outdata2, np.r_[outdata3, np.r_[outdata4, np.r_[outdata5, outdata6]]]]]
    print(outputdata.shape)

    # np.savetxt('../data/allInput.txt', inputdata, fmt='%10e %10e %10e %10e %10e %10e %10e %10e %10e %10e %10e %10e %10e'
    #                                                   ' %10e %10e %1e')
    np.savetxt('../data/allInput.txt', inputdata)
    np.savetxt('../data/allOutput.txt', outputdata)

if __name__ == '__main__':
    # inter_out()
    cutOUtIn()