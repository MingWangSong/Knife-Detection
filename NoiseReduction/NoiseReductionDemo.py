# 去噪

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 普通降噪方法
"""
均值模糊 : 去随机噪声有很好的去噪效果
（1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
"""
def blur_demo(image):
    dst = cv2.blur(image, (3, 3))
    # cv2.namedWindow("avg_blur_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("avg_blur_demo", dst)
    return dst

"""
中值模糊  对椒盐噪声有很好的去燥效果
"""
def median_blur_demo(image):
    dst = cv2.medianBlur(image, 3)
    # cv2.namedWindow("median_blur_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("median_blur_demo", dst)
    return dst

"""
用户自定义模糊
下面除以25是防止数值溢出
"""
def custom_blur_demo(image):
    kernel = np.ones([5, 5], np.float32)/25
    dst = cv2.filter2D(image, -1, kernel)
    # cv2.namedWindow("custom_blur_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("custom_blur_demo", dst)
    return dst

"""
    双边滤波
"""
def bi_demo(image):
    dst = cv2.bilateralFilter(image, 0, 100, 5)
    # cv2.namedWindow("bi_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("bi_demo", dst)
    return dst

"""
    均值迁移
"""
def shift_demo(image):
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    # cv2.namedWindow("shift_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("shift_demo", dst)
    return dst

"""
    高斯滤波
"""
def gaussian_demo(image):
    dst = cv2.GaussianBlur(image, (111, 111), 10, 0)
    # cv2.namedWindow("Gaussian_Blur2", cv2.WINDOW_NORMAL)
    # cv2.imshow("Gaussian_Blur2", dst)
    return dst

# def clamp(pv):
#     if pv > 255:
#         return 255
#     if pv < 0:
#         return 0
#     else:
#         return pv

"""
    方框滤波
"""
#方框滤波，normalize=1时，表示进行归一化处理，此时图片处理效果与均值滤波相同，如果normalize=0时，表示不进行归一化处理，像素值为周围像素之和，图像更多为白色
def boxfilter_demo(img):
    boxfilter = cv2.boxFilter(img, -1, (5, 5), normalize=1)
    return boxfilter

"""
    高通过滤/滤波（边缘检测/高反差保留）
    函数有：cv2.Sobel() , cv2.Schar() , cv2.Laplacian()
"""
def height_demo(img):

    # x轴或者y轴方向算子
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    dist = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    cv2.namedWindow("y", cv2.WINDOW_NORMAL)
    cv2.imshow('y', absy)
    cv2.namedWindow("x", cv2.WINDOW_NORMAL)
    cv2.imshow('x', absx)
    cv2.namedWindow("dsit", cv2.WINDOW_NORMAL)
    cv2.imshow('dsit', dist)

if __name__ == '__main__':
    # img = cv2.imread("./images/t1.png", 0)
    imageName="img7.png"
    imageAddress="../images/source/"
    img = cv2.imread(imageAddress+imageName, 0)
    # img = cv2.resize(src, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    # cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
    # cv2.imshow('input_image', src)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/gray.svg")

    mean_filter = blur_demo(img)
    plt.figure()
    plt.imshow(mean_filter, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/mean_filter.svg")

    median_filter = median_blur_demo(img)
    plt.figure()
    plt.imshow(median_filter, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/median_filter.svg")

    double_filter = bi_demo(img)
    plt.figure()
    plt.imshow(double_filter, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/double_filter.svg")

    gaussian_filter = gaussian_demo(img)
    plt.figure()
    plt.imshow(gaussian_filter, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/gaussian_filter.svg")

    boxFilter_filter = boxfilter_demo(img)
    plt.figure()
    plt.imshow(boxFilter_filter, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/boxFilter_filter.svg")




    # Filter_images = [img, mean_filter, median_filter, double_filter, gaussian_filter, boxFilter_filter]
    # # Filter_images = [src, mean_filter]
    # titles = [u'原图', u'均值滤波', u'中值滤波', u'双边滤波', u'高斯滤波', u'方框滤波']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.figure(figsize=(10, 7))
    # for i in range(len(Filter_images)):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(Filter_images[i], cmap="gray")
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    plt.show()