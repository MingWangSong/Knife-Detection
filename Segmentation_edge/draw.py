import cv2
from Segmentation_edge.Sobel import Sobel
from Segmentation_edge.Roberts import roberts_seg

# 读取图像
image_address = r'../images/source/img10.png'
img = cv2.imread(image_address, 0)  # 读入灰度图像

# 分割算法
canny_img = cv2.Canny(img, 100, 200)  # 参数为低阈值和高阈值
sobel_img = Sobel(image_address)
roberts_img = roberts_seg(image_address)

cv2.imwrite("canny_img.png", canny_img)
cv2.namedWindow("canny_img image", cv2.WINDOW_NORMAL)
cv2.imshow("canny_img image", canny_img)

cv2.imwrite("sobel_img.png", sobel_img)
cv2.namedWindow("sobel_img image", cv2.WINDOW_NORMAL)
cv2.imshow("sobel_img image", sobel_img)

cv2.imwrite("roberts_img.png", roberts_img)
cv2.namedWindow("roberts_img image", cv2.WINDOW_NORMAL)
cv2.imshow("roberts_img image", roberts_img)

cv2.waitKey()


# 显示结果
# plt.subplot(221), plt.imshow(img, cmap = 'gray')
# plt.title('附着物图像'), plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(otsu_img, cmap = 'gray')
# plt.title('OTSU分割法'), plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(max_img, cmap = 'gray')
# plt.title('最大熵分割法'), plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(die_img, cmap = 'gray')
# plt.title('阈值迭代分割法'), plt.xticks([]), plt.yticks([])
# plt.show()
