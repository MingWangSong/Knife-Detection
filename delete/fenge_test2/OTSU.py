# 导入库
import cv2
import time

# 读取图片
gray = cv2.imread('../../images/source/img10.png', 0)
# 调取opencv库自带的OTSU方法
t1 = time.time()
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
print("using time", time.time()-t1)
# 可视化 灰度图
cv2.imshow("gray image", gray)
# 可视化 阈值分割图
cv2.imwrite("OTSU.png", thresh)
cv2.namedWindow("threshold image", cv2.WINDOW_NORMAL)
cv2.imshow("threshold image", thresh)
cv2.waitKey()