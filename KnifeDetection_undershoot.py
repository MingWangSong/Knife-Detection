## 豁口检测
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import scipy.signal

def knife_detection(image_path):
    start_time = time.time()
    # 读取图像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.blur(gray_image, (3, 3))
    # 提取边缘二值化图
    edge = get_edge_by_prewitt(gray_image)
    # edge_twe_dimension = []
    # for i in range(len(edge)):
    #     temp = [i, edge[i]]
    #     edge_twe_dimension.append(np.asarray(temp))
    # result = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(result, np.asarray(edge_twe_dimension), -1, (0, 0, 255), 2)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 包络拟合
    envelope_y, peak_index, peak_num = envelope_by_three_interpolation(edge)
    # 豁口检测
    gap_start_index, gap_end_index = gap_detection(edge, envelope_y)
    # 将豁口范围扩充到最近峰值点
    gap_start_index, gap_end_index = gap_range_extend(peak_index, gap_start_index, gap_end_index)
    gap_start_index = list_distinc(gap_start_index)
    gap_end_index = list_distinc(gap_end_index)
    end_time = time.time()
    print("耗时: {:.4f}秒".format(end_time - start_time))
    # ---------------------------------------绘图--------------------------------------------
    # 用来正常显示中文标签，尺寸
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 单位是inches
    # 绘制背景图
    # img = plt.imread(image_path)
    # img = np.flipud(img)
    # plt.imshow(img)
    # 绘制边缘
    # plt.plot(edge, "-r", label='待测边缘')
    plt.plot(edge, "-r", label='Edge to be measured')
    # # 绘制峰值点
    # plt.scatter(peak_index, peak_num, c='g', marker='o', label='peak_num point')
    # 绘制包络线
    # plt.plot(envelope_y-1, ":b", label='包络拟合线')
    plt.plot(envelope_y-1, ":b", label='Envelope fitting line')

    # # 绘制豁口边界线
    # for start, end in zip(gap_start_index, gap_end_index):
    #     plt.axvline(start, c='green')
    #     plt.axvline(end, c='green')
    # 填充豁口区域
    # x = np.array(range(0, len(edge), 1))
    # for i in range(len(gap_start_index)):
    #     plt.fill_between(x, edge, envelope_y,
    #                      where=(edge <= envelope_y) & (gap_start_index[i] < x) & (x < gap_end_index[i]),
    #                      color='dodgerblue', alpha=0.5)
    # 绘图设置
    plt.xlim(0, gray_image.shape[1])
    plt.ylim(0, gray_image.shape[0])
    # plt.title('包络拟合示意图')
    # plt.title('基于四次Hermite插值的豁口检测')
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])  # 去y坐标刻度
    plt.legend()
    plt.show()

def list_distinc(nums):
    new_list = []
    for num in nums:
        if num not in new_list:
            new_list.append(num)
    return new_list

def get_edge_by_prewitt(gray_image):
    """
        提取灰度图边缘（Prewitt算子 + 边缘平滑）
    Args:
        gray_image: 灰度图像

    Returns:
        灰度图边缘
    """
    # Prewitt算子
    kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(gray_image, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gray_image, cv2.CV_16S, kernel_y)
    # 转uint8
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    prewitt = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    # 边缘图二值化
    ret, prewitt_binary = cv2.threshold(prewitt, 20, 255, cv2.THRESH_BINARY_INV)
    y_range, x_range = gray_image.shape
    edge = []
    for i in range(x_range):
        # 遍历每列，找出当前列所有0值所在行
        flag = False
        index_temp = []
        for j in range(y_range):
            # 第二个条件是防止噪声
            if prewitt_binary[j][i] == 0 and abs(j - y_range / 2) < y_range / 3:
                # 此处减法为了后续作图准备（图像原点在左上角，作图原点在左下角），此处适配转换
                index_temp.append(y_range - j)
                flag = True
            elif flag:
                flag = False
                break
        # 求行标平均值
        index = math.ceil(sum(index_temp) / len(index_temp))
        edge.append(index)
    # 边缘平滑（排除噪声影响）
    return scipy.signal.savgol_filter(edge, 53, 3)


def envelope_by_three_interpolation(edge):
    """
        基于三次样条插值的包络拟合
    Args:
        edge: 刀具边缘

    Returns:
        envelope_x：包络拟合曲线x
        envelope_y：包络拟合曲线y
        peak_x：检测的峰值索引
        peak_y: 检测的峰值高度
    """
    # 峰值检测函数，返回峰值在edge中的index
    peak_index = scipy.signal.argrelextrema(np.array(edge), comparator=np.greater, order=2)[0]
    # 计算峰值阈值
    peak_num = [edge[i] for i in peak_index]
    threshold = sum(peak_num) / len(peak_num)
    # 初始化三次样条插值，使得边缘避过起始点
    peak_x = [0]
    peak_y = [edge[0]]
    for index in peak_index:
        # 阈值筛选
        if edge[index] > threshold-7:
            peak_x.append(index)
            peak_y.append(edge[index])
    # 标记尾部点
    peak_x.append(len(edge) - 1)
    peak_y.append(edge[len(edge) - 1])
    # 三次样条插值
    tck = interpolate.splrep(peak_x, peak_y)
    envelope_x = list(range(0, len(edge), 1))
    envelope_y = interpolate.splev(envelope_x, tck, der=0)
    return envelope_y, peak_x, peak_y


def gap_detection(actual_edge, envelope_edge, gap_distance_threshold=40):
    """
        豁口检测
    Args:
        actual_edge: 采集的实际刀具边缘
        envelope_edge:  拟合的理论刀具边缘
        gap_distance_threshold:  豁口高度阈值

    Returns:
        gap_candidate_index_start：豁口起始索引
        gap_candidate_index_end：豁口结束索引

    """
    actual_edge = np.array(actual_edge)
    envelope_edge = np.array(envelope_edge)
    gap_distance = envelope_edge - actual_edge
    # 超过豁口距离阈值的索引
    gap_candidate_index = np.array([index for index in range(gap_distance.size)
                                    if gap_distance[index] >= gap_distance_threshold])
    # 搜索每个豁口结束位置
    gap_candidate_index_temp = np.append(gap_candidate_index, actual_edge.size - 1)
    # out[i] = a[i+1] - a[i]
    gap_candidate_index_diff = np.diff(gap_candidate_index_temp)
    gap_candidate_index_end = [i for i in range(gap_candidate_index_diff.size) if gap_candidate_index_diff[i] != 1]
    gap_candidate_index_end = gap_candidate_index_temp[gap_candidate_index_end]
    # 搜素每个豁口开始位置
    gap_candidate_index_temp = np.append(gap_candidate_index[::-1], 0)
    gap_candidate_index_diff = np.diff(gap_candidate_index_temp)
    gap_candidate_index_start = [i for i in range(gap_candidate_index_diff.size) if gap_candidate_index_diff[i] != -1]
    gap_candidate_index_start = gap_candidate_index_temp[gap_candidate_index_start]
    gap_candidate_index_start = gap_candidate_index_start[::-1]
    return gap_candidate_index_start, gap_candidate_index_end


def search_after(arr, target):
    """
        搜素arr有序数组中大于target最小的数
    Args:
        arr:     数组
        target:  目标数

    Returns:
        返回数值

    """
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return arr[low]


def search_before(arr, target):
    """
        搜素arr有序数组中小于target最大的数
    Args:
        arr:     数组
        target:  目标数

    Returns:
        返回数值

    """
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] > target:
            high = mid - 1
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid-1
    return arr[high]


def gap_range_extend(peak_index, gap_start_index, gap_end_index):
    """
       将豁口范围扩充到最近峰值点
    Args:
        peak_index:        峰值位置
        gap_start_index:   豁口起始位置
        gap_end_index:     豁口结束位置

    Returns:
        豁口扩充后的位置
    """
    for i in range(len(gap_start_index)):
        start_temp = gap_start_index[i]
        end_temp = gap_end_index[i]
        gap_start_index[i] = search_before(peak_index, start_temp)
        gap_end_index[i] = search_after(peak_index, end_temp)
    return gap_start_index, gap_end_index

# 图像处理函数，要传入路径
knife_detection(r'./images/source/basedata/huo/new/5.jpg')
# r'./images/source/basedata/huo/new/1.png'
# r'./images/source/img1.png'