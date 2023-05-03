# 图像二值化

import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    imageName = '571.png'
    sourceRoot = '../../images/source/'
    resultRoot = '../../images/result/'
    img = cv2.imread(sourceRoot + imageName, cv2.IMREAD_GRAYSCALE)
    _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    _, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    _, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    _, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    # 腐蚀图像
    thresh1 = cv2.erode(thresh1, kernel)
    thresh1 = cv2.erode(thresh1, kernel)
    # cv2.imwrite(resultRoot + imageName.split('.')[0] + "_Binarization.jpg", thresh1)

    titles = ['source Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(len(titles)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()