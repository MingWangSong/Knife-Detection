import cv2
import numpy as np
import Image
def roberts_seg(image_address):

    # 读取图像
    im = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)

    w, h = im.shape
    res = np.zeros((w, h))  # 取一个和原图一样大小的图片，并在里面填充0
    roberts_x = [[1, 0], [0, -1]]  # Roberts模板
    roberts_y = [[0, 1], [-1, 0]]
    for x in range(w - 1):
        for y in range(h - 1):
            sub = [[im.getpixel((x, y)), im.getpixel((x, y + 1))],
                   [im.getpixel((x + 1, y)), im.getpixel((x + 1, y + 1))]]  # x,y代表像素的位置，而不是像素值，要从图片上得到像素值
            sub = np.array(sub)  # 在python标准中list是不能做乘法，所以np.array()把list转就可以相乘
            roberts_x = np.array(roberts_x)
            roberts_y = np.array(roberts_y)
            var_x = sum(sum(sub * roberts_x))
            # 矩阵相乘，查看公式，我们要得到是一个值，所以对它进行两次相加
            var_y = sum(sum(sub * roberts_y))

            var = abs(var_x) + abs(var_y)

            res[x][y] = var
            # 把var值放在x行y列位置上

        return res