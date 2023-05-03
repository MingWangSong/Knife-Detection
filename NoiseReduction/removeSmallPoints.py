"""
python-opencv去除孤立点
skimage.__version__==0.19.1
"""
import numpy as np
from skimage import morphology, color
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


def remove_small_points(binary_img, threshold_area):
    """
    消除二值图像中面积小于某个阈值的连通域(消除孤立点)
    args:
        binary_img: 二值图
        threshold_area: 面积条件大小的阈值,大于阈值保留,小于阈值过滤
    return:
        resMatrix: 消除孤立点后的二值图
    """
    # 输出二值图像中所有的连通域 img_label代表每个区域不同的标记
    img_label, num = label(binary_img, connectivity=1, background=255, return_num=True)  # connectivity=1--4  connectivity=2--8
    # 绘制连通区域
    # cells_color = color.label2rgb(img_label, bg_label=0, bg_color=(255, 255, 255))
    # plt.figure()
    # plt.imshow(cells_color)
    # print('+++', num, img_label)
    # 输出连通域的属性，包括面积等
    props = regionprops(img_label)
    ## adaptive threshold
    resMatrix = np.full(img_label.shape, 255).astype(np.uint8)
    # resMatrix = np.zeros(img_label.shape).astype(np.uint8)
    for i in range(0, len(props)):
        print('--', props[i].area)
        if props[i].area > threshold_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            # 组合所有符合条件的连通域
            tmp *= 255
            resMatrix -= tmp
    return resMatrix


if __name__ == '__main__':
    import cv2

    ##png可以保存二值图, jpg保存的是灰度图
    img_pth = "../images/source/img7.png"
    ##读入彩色图片
    # img = cv2.imread(img_pth)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##读入灰度图片
    img_gray = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE)
    # fixed thresh
    _, thresh = cv2.threshold(img_gray, 127, 255, type=cv2.THRESH_BINARY)
    ## method1
    thresh1 = thresh > 0
    # res_img1 = morphology.remove_small_objects(thresh1, 2)
    # res_img0 = morphology.remove_small_holes(thresh1, 20)
    ## method2
    res_img2 = remove_small_points(thresh, 1000)
    plt.imshow(res_img2, cmap='gray')
    plt.axis('off')
    plt.savefig("../images/result/ResulrtnoitceReduction/removeSmall.svg")
    plt.show()

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title('ori img')
    # plt.imshow(img_gray, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.title('method1')
    # plt.imshow(res_img1, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.title('method2')
    # plt.imshow(res_img2, cmap='gray')
    # plt.show()