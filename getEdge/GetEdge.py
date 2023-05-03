# 边缘轮廓提取

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# 边缘轮廓坐标提取函数
# 拟合辅助函数
def func(x, a, b, c):
  return b * np.power(a, x) + c

# 提取边缘函数
def getEdgeByImage(bgr_img):
    #读取图片，该图在此代码的同级目录下
    #bgr_img = cv2.imread("./images/img3.png")
    # BGR转灰度图
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    #二化值
    th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)

    #获取轮廓的点集，cv2.findContours()函数接受的参数为二值图
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #取最大的边缘轮廓点集
    contours = max(contours, key = cv2.contourArea)

    #求取轮廓的矩
    # M = cv2.moments(contours)

    #画出轮廓
    cv2.drawContours(bgr_img, contours, -1, (255, 0, 255), 10)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    #在图片上画出矩形边框
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 获取曲线坐标
    contours_reshape = np.reshape(contours, (-1, 2))
    x = contours_reshape[1:-2, 0]
    y = contours_reshape[1:-2, 1]
    temp = {
        'x': pd.Series(x),
        'y': pd.Series(y)
    }
    df = pd.DataFrame(temp)
    # 根据x值去重，避免一个x值对应多个y值
    df.drop_duplicates(subset=['x'], keep='first', inplace=True)
    return np.array(df['x']), np.array(df['y'])

# 绘图函数
def show(x, y):
    # 最小二乘曲线拟合
    popt, pcov = curve_fit(func, x, y)
    y_pred = [func(i, popt[0], popt[1], popt[2]) for i in x]
    plt.plot(x, y)
    plt.plot(x, y_pred, 'r', label='fit values')
    plt.xlim(0, 4052)
    plt.ylim(-500, 2500)
    #获取到当前坐标轴信息
    ax = plt.gca()
    #将X坐标轴移到上面
    ax.xaxis.set_ticks_position('top')
    #反转Y坐标轴
    ax.invert_yaxis()
    plt.show()
    # 绘制轮廓
    # cv2.namedWindow("name", cv2.WINDOW_NORMAL)
    # cv2.imshow("name", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


bgr_img = cv2.imread("../images/source/img3.png")
x, y = getEdgeByImage(bgr_img)
# 求正则项，即平均值
c = np.mean(y)
y = y - c
show(x, y)