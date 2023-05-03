# 检测思路
# 1、二值化获取白色亮区域，接下来转为检测白色区域的凹缺陷来检测
# 2、使用半径为n的圆形结构元素对原图做闭运算（n的确定由检测精度决定）此步骤可达到平滑的效果
# 3、二值化结果区域与闭运算结果区域做差
# 4、5x5的结构元素做开运算，滤除边缘噪点，剩余真正的缺陷区域
import numpy as np
import cv2
import datetime

# 存图
write = True
# write = False
image_name = "img10.png"
images_address = r'../images/source/litter/'+image_name
gray = cv2.imread(images_address, cv2.IMREAD_GRAYSCALE)
size = gray.shape
# cv2.namedWindow('gray', 0)
# cv2.resizeWindow('gray', size[1], size[0])
cv2.imshow('gray', gray)

# OTSU阈值二值化
ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.namedWindow('threshold', cv2.WINDOW_FREERATIO)
cv2.imshow('threshold', threshold)
if write:
  now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  cv2.imwrite(r"../images/source/litter/res/threshold_"+ now_time+ "_" +image_name, threshold)

# 250*250圆形结构元素--闭运算--平滑处理(闭运算平滑黑色边缘，开运算平滑白色边缘)
k1 = np.zeros((220, 220), np.uint8)
k_size = k1.shape
cv2.circle(k1, (int(k_size[0]/2), int(k_size[1]/2)), int(k_size[0]/2), (1, 1, 1), -1, cv2.LINE_AA)
closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, k1, None, None, 1)
# cv2.namedWindow('closing', cv2.WINDOW_FREERATIO)
cv2.imshow('closing', closing)

# 图像差分（获取附着物部分）
diff = cv2.absdiff(threshold, closing)
cv2.imshow('diff', diff)

# 5*5圆形结构元素--开运算（去除图像中的校白点）
# k2=np.ones((5,5), np.uint8) #矩形结构元素
# k2 = np.zeros((5, 5), np.uint8)
# cv2.circle(k2, (2, 2), 2, (1, 1, 1), -1, cv2.LINE_AA)
# opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k2)
# cv2.imshow('opening', opening)

# 方法二:降噪处理会使得图片模糊
# opening = cv2.blur(diff, (5, 5))

# 方法三：差分后小连通区域去除
diff_new = 255-diff
# cv2.imshow('diff_new', diff_new)
contours, _ = cv2.findContours(diff_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)  # 轮廓的个数
cv_contours = []
for contour in contours:
  area = cv2.contourArea(contour)
  if area <= 350:
    cv_contours.append(contour)
  else:
    continue
cv2.fillPoly(diff_new, cv_contours, (255, 255, 255))
opening = 255-diff_new
# cv2.namedWindow('opening', cv2.WINDOW_FREERATIO)
cv2.imshow('opening', opening)
if write:   
  now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  cv2.imwrite(r"../images/source/litter/res/opening_"+ now_time+ "_" +image_name, opening)
# 轮廓查找
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

isNG = False
# 结果判断(还可以自己设置缺陷大小来删选)
if len(contours) > 0:
  isNG = True
  # 在二值化图上框定
  result = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
  cv2.drawContours(result, contours, -1, (0, 0, 255), 5)
  # 在原图上框定gray
  # result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
  # cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

# threshold图打NG标签
# if isNG:
#   rect, basline = cv2.getTextSize('Detect NG', cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
#   cv2.rectangle(threshold, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
#   cv2.putText(threshold,'Detect NG', (10,5+rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
# else:
#   rect, basline = cv2.getTextSize('Detect OK', cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
#   cv2.rectangle(threshold, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
#   cv2.putText(threshold,'Detect OK', (10,5+rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

# 显示result结果
cv2.imshow('result', result)
if False:
  now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  cv2.imwrite(r"../images/source/litter/res/result_"+ now_time+ "_" +image_name,result)
cv2.waitKey(0)
cv2.destroyAllWindows()