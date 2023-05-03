# cv2.cv2 用于代码提示
import cv2 as cv
import numpy as np
# 图像输入
input_img = cv.imread('../images/source/noclean-001.png', 0)
# 降噪处理
blur = cv.blur(input_img, (3, 3), 0)
# cv.imshow("blur",blur)
# 图像二值化
# thre, thres = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV)
_, thres = cv.threshold(blur, 180, 255, cv.THRESH_BINARY_INV)
cv.namedWindow("image_thre", 0)
cv.imshow("image_thre", thres)
# 形态学开运算
kernel = np.ones((3, 3), np.uint8)
img_bin = cv.morphologyEx(thres, cv.MORPH_OPEN, kernel)
cv.namedWindow("image_bin", 0)
cv.imshow("image_bin", img_bin)
# 轮廓提取
# contours:轮廓坐标列表
# hierarchy：轮廓从属关系
contours, hierarchy = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#计算凸包数量
for i in range(len(contours)):
    cnt =contours[i]
    # 返回该物体得凸包点集
    hull = cv.convexHull(cnt, returnPoints=False)
    print(hull)
    if len(hull) > 10:
        # 凸缺陷计算
        defects = cv.convexityDefects(cnt, hull)
        if len(defects) >= 2:
            # 升序排列
            defects_point = defects[np.lexsort(-defects.T)]
            # 起点，终点，最远点，到最远点的近似距离
            os, oe, of, od = defects_point[0, 0, 0]
            ostart = tuple(cnt[of][0])
            ts, te, tf, td = defects_point[0, 1, 0]
            tstart = tuple(cnt[tf][0])
            if (od/256)>10:
                cv.line(img_bin, tuple(ostart), tuple(tstart), [0, 0, 255], 2)

cv.namedWindow("image_bin_end", 0)
cv.imshow("image_bin_end", img_bin)

# #再次找轮廓花外接矩形
# contours2 ,hierarchy2  = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# for c in contours2:
#     #拿到该轮廓得最小矩形得顶点坐标和长宽
#     x, y, w, h = cv.boundingRect(c)
#     cv.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 255), 1)
#     #计算圆心位置
#     cx = int(x + w / 2)
#     cy = int(y + h / 2)
#     cv.circle(input_img, (cx, cy), 1, (0, 0, 255), -1)  # 用圆点绘制目标重心
# cv.namedWindow("image_bin_line", 0)
# cv.imshow("image_bin_line", input_img)
cv.waitKey()
