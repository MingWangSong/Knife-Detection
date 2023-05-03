## 凹点检测

import cv2
import numpy as np

## 凹点检测
def detection(img_detect):
    # 轮廓提取  输入的图片必须是二值图片
    contours, hierarchy = cv2.findContours(img_detect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img_detect_show = cv2.resize(img_detect, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)  # 比例因子：fx=0.5,fy=0.5
    #cv2.imshow("img_detect", img_detect_show)
    # 计算凸包数量
    for i in range(len(contours)):
        cnt = contours[i]
        min_dist = 0
        # 返回该物体得凸包个数
        hull = cv2.convexHull(cnt, returnPoints=False)  # 需要point2d
        if len(hull) > 10:
            # 缺陷计算
            defects = cv2.convexityDefects(cnt, hull)
            if len(defects) >= 2:
                # 按第四个之倒叙排序
                defects_point = defects[np.lexsort(-defects.T)]  # 多列数据排序，优先照顾后面的列
                o_s, oe, of, od = defects_point[0, 0, 0]
                ostart = tuple(cnt[of][0])
                ts, te, tf, td = defects_point[0, 1, 0]
                tstart = tuple(cnt[tf][0])
                if (od / 256) > 10:
                    cv2.line(img_detect, tuple(ostart), tuple(tstart), [0, 0, 255], 5)
    return img_detect

if __name__ == '__main__':
    img = cv2.imread("../images/source/noclean-001.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    res = detection(thresh)
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.imshow("res", res)
    cv2.waitKey(0)