# 分水岭算法
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import segmentation,feature
import cv2
#新增加的两行
import matplotlib
matplotlib.rc("font", family='FangSong')

#创建两个带有重叠圆的图像
# x, y = np.indices((80, 80))
# x1, y1, x2, y2 = 28, 28, 44, 52
# r1, r2 = 16, 20
# mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
# mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
# image = np.logical_or(mask_circle1, mask_circle2)

image = cv2.imread('../images/source/noclean-028_litter.png')
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#现在我们用分水岭算法分离两个圆
distance = ndi.distance_transform_edt(image) #距离变换
local_maxi =feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=image)   #寻找峰值
markers = ndi.label(local_maxi)[0]           #初始标记点
labels =segmentation.watershed(-distance, markers, mask=image) #基于距离变换的分水岭算法
# kimage.segmentation.watershed


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))


# plt.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')

plt.imshow(labels, cmap=plt.cm.gray, interpolation='nearest')



# cv2.imwrite("distance.png", -distance)
# cv2.imwrite("markers.png", markers)
# cv2.imwrite("labels.png", labels)


# for ax in axes:
#     ax.axis('off')
# fig.tight_layout()
plt.axis('off')
plt.show()