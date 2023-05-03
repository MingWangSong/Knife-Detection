# 去噪

import matplotlib.pyplot as plt
import numpy as np
import cv2

# 高级降噪方法

# 图像预处理
def log_filter(gray_img):
    # 高斯滤波  以核大小为3x3，方差为0
    gaus_img = cv2.GaussianBlur(gray_img,(3,3),sigmaX=0)
    # Laplacian算子检测边缘
    log_img = cv2.Laplacian(gaus_img,cv2.CV_16S,ksize=3)
    # 图像增强
    log_img = cv2.convertScaleAbs(log_img)
    return log_img

# 各种卷积核
def filter_imgs(gray_img):
    # 尝试一下不同的核的效果
    Emboss = np.array([[ -2,-1, 0],
                       [ -1, 1, 1],
                       [  0, 1, 2]])

    Motion = np.array([[ 0.333, 0,  0],
                       [  0, 0.333, 0],
                       [  0, 0, 0.333]])

    Emboss_img = cv2.filter2D(gray_img,cv2.CV_16S,Emboss)
    Motion_img = cv2.filter2D(gray_img, cv2.CV_16S, Motion)
    Emboss_img = cv2.convertScaleAbs(Emboss_img)
    Motion_img = cv2.convertScaleAbs(Motion_img)

    different_V = np.array([[  0, -1, 0],
                            [  0,  1, 0],
                            [  0,  0, 0]])
    different_H = np.array([[  0, 0, 0],
                            [ -1, 1, 0],
                            [  0, 0, 0]])
    different_temp = cv2.filter2D(gray_img,cv2.CV_16S,different_V)
    different_temp = cv2.filter2D(different_temp, cv2.CV_16S, different_H)
    different_img = cv2.convertScaleAbs(different_temp)

    Sobel_V = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [ -1, -2, -1]])
    Sobel_H = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    Sobel_temp = cv2.filter2D(gray_img,cv2.CV_16S, Sobel_V)
    Sobel_temp = cv2.filter2D(Sobel_temp, cv2.CV_16S, Sobel_H)
    Sobel_img = cv2.convertScaleAbs(Sobel_temp)


    Prewitt_V = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])
    Prewitt_H = np.array([[-1,  0, 1],
                          [-1,  0, 1],
                          [-1,  0, 1]])
    Prewitt_temp = cv2.filter2D(gray_img, cv2.CV_16S, Prewitt_V)
    Prewitt_temp = cv2.filter2D(Prewitt_temp, cv2.CV_16S, Prewitt_H)
    Prewitt_img = cv2.convertScaleAbs(Prewitt_temp)

    kernel_P = np.array([[0,  0, -1,  0, 0],
                         [0, -1, -2, -1, 0],
                         [-1,-2, 16, -2,-1],
                         [0, -1, -2, -1, 0],
                         [0, 0,  -1, 0,  0]])
    kernel_N = np.array([[0, 0,  1,  0, 0],
                         [0, 1,  2,  1, 0],
                         [1, 2, -16, 2, 1],
                         [0, 1,  2,  1, 0],
                         [0, 0,  1,  0, 0]])


    lap4_filter = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])  # 4邻域laplacian算子
    lap8_filter = np.array([[0, 1, 0],
                            [1, -8, 1],
                            [0, 1, 0]])  # 8邻域laplacian算子
    lap_filter_P = cv2.filter2D(gray_img, cv2.CV_16S, kernel_P)
    edge4_img_P = cv2.filter2D(lap_filter_P, cv2.CV_16S, lap4_filter)
    edge4_img_P = cv2.convertScaleAbs(edge4_img_P)
    # 卷积
    edge8_img_P = cv2.filter2D(lap_filter_P, cv2.CV_16S, lap8_filter)
    edge8_img_P = cv2.convertScaleAbs(edge8_img_P)


    lap_filter_N = cv2.filter2D(gray_img, cv2.CV_16S, kernel_N)
    edge4_img_N = cv2.filter2D(lap_filter_N, cv2.CV_16S, lap4_filter)
    edge4_img_N = cv2.convertScaleAbs(edge4_img_N)

    edge8_img_N = cv2.filter2D(lap_filter_N, cv2.CV_16S, lap8_filter)
    edge8_img_N = cv2.convertScaleAbs(edge8_img_N)
    return (Emboss_img,Motion_img,different_img,Sobel_img,Prewitt_img,edge4_img_P,edge8_img_P,edge4_img_N,edge8_img_N)

# 绘图
def show(Filter_imgs):
    titles = [u'原图',     u'Laplacian算子',\
              u'Emboss滤波',u'Motion滤波',
              u'diff(差分)滤波',u'Sobel滤波',u'Prewitt滤波',
              u'Lap4算子-kernel_P', u'Lap8算子-kernel_P',
              u'Lap4算子-kernel_N', u'Lap8算子-kernel_N']

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 8))
    for i in range(len(titles)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(getEdge(Filter_imgs[i]), plt.cm.gray)
        #plt.imshow(Filter_imgs[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def getEdge(img):
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    gaussImg = cv2.Canny(blurred, 10, 70)
    return gaussImg

if __name__ == '__main__':
    img = cv2.imread('../images/source/img1.png')
    # 转RGB图
    img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转单通道灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像预处理（滤波平滑+边缘检测+图像增强）
    LoG_img = log_filter(gray_img)
    Filter_imgs = [img_raw,LoG_img]
    # 各种卷积
    Filter_imgs.extend(filter_imgs(gray_img))
    show(Filter_imgs)
