## 边缘提取

import cv2
import numpy as np
import matplotlib.pyplot as plt

def put(path):
    # 读取图像
    img = cv2.imread(path)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)

    # 二值化
    ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    # Sobel算子
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 拉普拉斯算子
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)

    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(grayImage, (5, 5), 0)

    # Canny算子
    Canny = cv2.Canny(gaussian, 50, 150)

    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 高斯拉普拉斯算子
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)  # 先通过高斯滤波降噪
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)  # 再通过拉普拉斯算子做边缘检测
    LOG = cv2.convertScaleAbs(dst)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图形
    # plt.subplot(241), plt.imshow(img2), plt.title('原始图像'), plt.axis('off')
    # plt.subplot(242), plt.imshow(binary, plt.cm.gray), plt.title('二值图'), plt.axis('off')
    # plt.subplot(243), plt.imshow(Sobel, plt.cm.gray), plt.title('Sobel算子'), plt.axis('off')
    # plt.subplot(244), plt.imshow(Roberts, plt.cm.gray), plt.title('Roberts算子'), plt.axis('off')
    # plt.subplot(245), plt.imshow(Laplacian, plt.cm.gray), plt.title('拉普拉斯算子'), plt.axis('off')
    # plt.subplot(246), plt.imshow(Canny, plt.cm.gray), plt.title('Canny算子'), plt.axis('off')
    # plt.subplot(247), plt.imshow(LOG, plt.cm.gray), plt.title('高斯拉普拉斯算子'), plt.axis('off')
    # plt.subplot(248), plt.imshow(Prewitt, plt.cm.gray), plt.title('Prewitt算子'), plt.axis('off')

    plt.figure(), plt.imshow(img2), plt.axis('off')
    plt.figure(), plt.imshow(binary, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(Sobel, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(Roberts, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(Laplacian, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(Canny, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(LOG, plt.cm.gray),  plt.axis('off')
    plt.figure(), plt.imshow(Prewitt, plt.cm.gray),  plt.axis('off')


    # plt.savefig('1.new-2.jpg')
    plt.show()

# 图像处理函数，要传入路径
put(r'../images/source/img13_new.jpg')