# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2

# 我们已成功检测到图像中的所有九个硬币。此外，我们还能够清晰地绘制每个硬币周围的边界。
# # 这与使用简单阈值检测和轮廓检测的先前示例形成了鲜明对比，在先前示例中，仅（错误地）检测到两个对象。
image = cv2.imread('../images/source/img10.png')
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.imshow("Input", image)
# 应用金字塔均值漂移滤波 以提高阈值设置步骤的准确性  【金字塔均值偏移滤波可以看做是对彩色图像平滑颜色的一种操作】
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.namedWindow("shifted", cv2.WINDOW_NORMAL)
cv2.imshow("shifted", shifted)

# 将经过金字塔均值偏移滤波处理的图像 读取为灰度图像
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
# 应用Otsu的阈值将背景从前景中分割出来：
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
cv2.imshow("Thresh", thresh)

# 分割的第一步：通过distance_transform_edt计算欧几里德距离变换（EDT Euclidean distance）功能
# 此函数为每个前景像素计算最接近零的欧几里得距离（即背景像素）。
## 计算所有非0像素点到最近0点的距离
D = ndimage.distance_transform_edt(thresh)
# 在距离图中找到峰值（即局部最大值）。我们将确保每个峰之间的距离至少为20像素。
# # 采用peak_local_max的输出功能，并使用8连接性应用连接组件分析。
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

# 该函数的输出为我们提供了标记 然后我们将其馈入分水岭函数
# 分水岭算法  假设我们的标记代表我们的距离图中的局部最小值（即山谷），因此我们采用D的负值。
# 分水岭函数返回标签矩阵，一个NumPy数组，其宽度和高度与我们的输入图像相同。每个像素值作为唯一的标签值。具有相同标签值的像素属于同一对象。
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# 最后一步是简单地循环唯一标签值并提取每个唯一对象
# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
    # label为0 默认为背景，忽略
    if label == 0:
        continue
    # 为我们的遮罩分配内存 并将属于当前标签的像素设置为255（白色）。
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    # 检测到遮罩中的轮廓   并提取最大的轮廓-该轮廓将代表图像中给定对象的轮廓/边界。
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # 绘制围绕对象的包围圆边界。我们还可以计算对象的边界框，应用按位运算，并提取每个单独的对象。
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", image)
cv2.waitKey(0)