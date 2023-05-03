# 图像二值化

import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    imageName = '571.png'
    sourceRoot = '../../images/source/'
    resultRoot = '../../images/result/'
    img = cv2.imread(sourceRoot + imageName, cv2.IMREAD_GRAYSCALE)
    _, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    plt.imshow(thresh4, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()