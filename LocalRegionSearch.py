from server.server.ins.util import RasancFitCircle as rasan, DrawSector as ds
import cv2
import numpy as np
import math


def pointerMaskBySector(areas, gray, center, patch_degree, radius):
    """
    用扇形遮罩的方法求直线位置。函数接受一个已经被灰度/二值化的图，图中默认保留了比较清晰的指针轮廓，
    该方法将圆的表盘分成360 / patch_degree 个区域，每个区域近似一个扇形，计算每个扇形面积的灰度和，
    然后在所有扇形区域中取出面积最大的那一个，如果处理预处理妥当，指针应该位于灰度值最大的区域.算法的
    精度决定于patch_degree的大小
    :param areas: 每个遮罩取出的区域
    :param gray: 灰度图
    :param center:圆的中心
    :param patch_degree:每个扇形区域所占的角度,该值越小，产生的遮罩越多，获取到的区域越细小
    :param radius: 圆的半径
    :return: 以(index,sum)形式组织的有序列表，index是扇形递增的序号，即每个扇形所在的区域向量与水平线夹角为index * patch_degree度
    """
    mask_res = []
    patch_index = 0
    masks, mask_centroids = ds.buildCounterClockWiseSectorMasks(center, radius, gray.shape, patch_degree,
                                                                (255, 0, 0),
                                                                reverse=True)
    for mask in masks:
        and_mask = cv2.bitwise_and(mask, gray)
        areas.append(and_mask)
        # mask_res.append((patch_index, np.sum(and_mask), and_mask))
        mask_res.append((patch_index, np.sum(and_mask)))
        patch_index += 1
    mask_res = sorted(mask_res, key=lambda r: r[1], reverse=True)
    return mask_res, mask_res[0][1] * patch_degree


def findPointerFromBinarySpace(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5):
    """
    接收一张预处理过的二值图（默认较完整保留了指针信息），从通过圆心水平线右边的点开始，连接圆心顺时针建立直线遮罩，取出遮罩范围下的区域,
    计算对应区域灰度和，灰度和最大的区域即为指针所在的位置。直线遮罩的粗细程度、搜索的梯度决定了算法侦测指针的细粒度。该算法适合搜索指针形状
    为直线的仪表盘，原理与@pointerMaskBySector类似。
    :param radians_low:圆的搜索范围(弧度制表示)
    :param radians_high:圆的搜索范围(弧度制表示)
    :param src: 二值图
    :param center: 刻度盘的圆心
    :param radius: 圆的半径
    :param patch_degree:搜索梯度，默认每次一度
    :param ptr_resolution: 指针的粗细程度
    :return: 指针遮罩、直线与圆相交的点
    """
    _shape = src.shape
    img = src.copy()
    # 弧度转化为角度值
    low = math.degrees(radians_low)
    high = math.degrees(radians_high)
    # _img1 = cv2.erode(_img1, kernel3, iterations=1)
    # _img1 = cv2.dilate(_img1, kernel3, iterations=1)
    # 157=pi/2*100
    mask_info = []
    max_area = 0
    best_theta = 0
    iteration = np.abs(int((high - low) / patch_degree))
    for i in range(iteration):
        # 建立一个大小跟输入一致的全黑图像
        # pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
        # theta = float(i) * 0.01
        # 每次旋转patch_degree度，取圆上一点
        theta = float(i * patch_degree / 180 * np.pi)
        pointer_mask, point = drawLineMask(_shape, theta, center, ptr_resolution, radius)
        # cv2.circle(black_img, (x1, y1), 2, 255, 3)
        # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
        # cv2.line(pointer_mask, (center[0], center[1]), point, 255, ptr_resolution)
        # 去除遮罩对应的小区域
        and_img = cv2.bitwise_and(pointer_mask, img)
        not_zero_intensity = cv2.countNonZero(and_img)
        mask_info.append((not_zero_intensity, theta))
        # if not_zero_intensity > mask_intensity:
        #     mask_intensity = not_zero_intensity
        #     mask_theta = theta
        # imwrite(dir_path+'/2_line1.jpg', black_img)
    # 按灰度和从大到小排列
    mask_info = sorted(mask_info, key=lambda m: m[0], reverse=True)
    # thresh = mask_info[0][0] / 30
    # over_index = 1
    # sum = thresh
    # for info in mask_info[1:]:
    #     if mask_info[0][0] - info[0] > thresh:
    #         break
    #     over_index += 1
    best_theta = mask_info[0][1]
    # 得到灰度和最大的那个直线遮罩,和直线与圆相交的点
    pointer_mask, point = drawLineMask(_shape, best_theta, center, ptr_resolution, radius)
    #
    # black_img1 = np.zeros([_shape[0], _shape[1]], np.uint8)
    # r = item[2]-20 if item[2]==_heart[1][2] else _heart[1][2]+ _heart[0][1]-_heart[1][1]-20
    # y1 = int(item[1] - math.sin(mask_theta) * (r))
    # x1 = int(item[0] + math.cos(mask_theta) * (r))
    # cv2.line(black_img1, (item[0], item[1]), (x1, y1), 255, 7)
    # src = cv2.subtract(src, line_mask)
    # img = cv2.subtract(img, line_mask)
    best_theta = 180 - best_theta * 180 / np.pi
    if best_theta < 0:
        best_theta = 360 - best_theta
    return pointer_mask, best_theta, point


def figureOutDialCircleByScaleLine(contours, dst_threshold, iter_time,
                                   period_rasanc_time):
    """
    无圆心、半径标定情况，根据刻度线拟合出表盘的圆模型
    :param contours: 刻度轮廓
    :param dst_threshold: 被视为inliers的阈值
    :param iter_time: 执行rasanc算法的次数
    :param period_rasanc_time: 每趟rasanc 的迭代次数
    :return: 拟合得到的圆心、半径
    """
    avg_circle = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    avg_fit_num = 0
    hit_time = 0
    centroids = []
    for contour in contours:
        mu = cv2.moments(contour)
        if mu['m00'] != 0:
            centroids.append((mu['m10'] / mu['m00'], mu['m01'] / mu['m00']))
    # for centroid in centroids:
    #     # rgb_src[int(centroid[0]), int(centroid[1])] = (0, 255, 0)
    #     r = np.random.randint(0, 256)
    #     g = np.random.randint(0, 256)
    #     b = np.random.randint(0, 256)

    # 为了确保拟合所得的圆的确信度，多次拟合求平均值
    for i in range(iter_time):
        best_circle, max_fit_num, best_consensus_pointers = rasan.randomSampleConsensus(centroids,
                                                                                        max_iterations=period_rasanc_time,
                                                                                        dst_threshold=dst_threshold,
                                                                                        inliers_threshold=len(
                                                                                            centroids) / 2,
                                                                                        optimal_consensus_num=int(
                                                                                            len(centroids) * 0.8))
        if max_fit_num > 0:
            hit_time += 1
            avg_circle += best_circle
            avg_fit_num += max_fit_num
    if hit_time > 0:
        avg_circle /= hit_time
        avg_fit_num /= hit_time
    # 求平均值减少误差
    center = (np.int(avg_circle[0]), np.int(avg_circle[1]))
    radius = np.int(avg_circle[2])
    if avg_fit_num > len(centroids) / 2:
        return center, radius
    else:
        print("Fitting Circle Failed.")
        return (0, 0), 0


def drawLineMask(_shape, best_theta, center, ptr_resolution, radius):
    """
    画一个长为radius，白色的直线，产生一个背景全黑的白色直线遮罩
    :param _shape:
    :param best_theta:
    :param center:
    :param ptr_resolution:
    :param radius:
    :return:
    """
    pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
    y1 = int(center[1] - np.sin(best_theta) * radius)
    x1 = int(center[0] + np.cos(best_theta) * radius)
    cv2.line(pointer_mask, (center[0], center[1]), (x1, y1), 255, ptr_resolution)
    return pointer_mask, (x1, y1)
