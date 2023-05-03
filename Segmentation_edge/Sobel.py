import cv2

def Sobel(image_address):
    img = cv2.imread(image_address, 0)  # 读入灰度图像

    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 参数分别为图像、数据类型、x方向导数、y方向导数
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 转换为8位灰度图像
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并梯度图像
    grad_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad_img

