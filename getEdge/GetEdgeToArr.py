## 边缘提取转换成曲线数据(最终版)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy.signal
import peakutils.peak

def put(path):
    # 读取图像
    grayImage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    grayImage = cv2.blur(grayImage, (3, 3))

    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 边缘图二值化
    ret, Prewitt_binary = cv2.threshold(Prewitt, 20, 255, cv2.THRESH_BINARY_INV)
    # 提取边缘坐标(高度y值存储在xy中)
    y_range, x_range = grayImage.shape
    xy = []
    for i in range(x_range):
        # 遍历每列，找出当前列0值所在行
        flag = False
        indexTemp = []
        for j in range(y_range):
            # 第二个条件是防止噪声
            if Prewitt_binary[j][i] == 0 and abs(j - y_range/2) < y_range/3:
                # 此处减法为了后续作图准备（图像原点在左上角，作图原点在左下角），此处适配转换
                indexTemp.append(y_range - j)
                flag = True
            elif flag:
                break
        # 求行标平均值
        index = math.ceil(sum(indexTemp) / len(indexTemp))
        xy.append(index)
        flag = False
    # 边缘平滑（排除噪声影响）
    xy = scipy.signal.savgol_filter(xy, 53, 3)
    # 峰值检测函数
    indexes = scipy.signal.argrelextrema(
        np.array(xy),
        comparator=np.greater, order=2
    )[0]
    # 计算峰值阈值
    peakNums = [xy[i] for i in indexes]
    threshold = sum(peakNums) / len(peakNums)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    plt.subplot(221), plt.imshow(grayImage, plt.cm.gray), plt.title('原始图'), plt.yticks([]), plt.xticks([])
    # plt.subplot(222), plt.imshow(Prewitt, plt.cm.gray), plt.title('Prewitt算子'), plt.yticks([]), plt.xticks([])
    # plt.subplot(223), plt.imshow(Prewitt_binary, plt.cm.gray), plt.title('Prewitt算子提取边缘二值化图'), plt.yticks([]), plt.xticks([])
    plt.subplot(222), plt.imshow(Prewitt, plt.cm.gray), plt.title('改进的Zernike矩边缘提取结果图'), plt.yticks([]), plt.xticks([])
    plt.subplot(223), plt.imshow(Prewitt_binary, plt.cm.gray), plt.title('边缘二值化图'), plt.yticks([]), plt.xticks([])
    ax = plt.subplot(224)
    ax.plot(xy, "-r")
    ## 初始化三次样条插值，使得边缘避过起始点
    splrep_x = [0]
    splrep_y = [xy[0]]
    ax.plot(0, xy[0], c='g', marker='x')
    for i in indexes:
        # 阈值筛选
        if xy[i] > threshold:
            splrep_x.append(i)
            splrep_y.append(xy[i])
            ax.plot(i, xy[i], c='g', marker='x')
    # 标记尾部点
    ax.plot(len(xy) - 1, xy[len(xy) - 1], c='g', marker='x')
    splrep_x.append(len(xy)-1)
    splrep_y.append(xy[len(xy)-1])
    # 三次样条插值
    tck = interpolate.splrep(splrep_x, splrep_y)
    xx = list(range(1, x_range, 1))
    yy = interpolate.splev(xx, tck, der=0)
    ax.plot(xx, yy, "b:")
    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    plt.title('边缘峰值检测图')
    plt.show()

# 图像处理函数，要传入路径
put(r'../images/source/noclean-001.png')
