import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances
from DebugSwitcher import is_debugging
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import os


def meterFinderByTemplate(image, template):
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox image
    """
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    w, h, _ = template.shape

    # for test
    # cv2.imshow("test", img)
    # img = (img * 0.5).astype(np.uint8) # test
    # cv2.imshow("test2", img)
    # cv2.waitKey(0)

    i = 5
    res = cv2.matchTemplate(image, template, methods[i])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    if methods[i] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + h, topLeft[1] + w)

    return image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]


def meterFinderBySIFT(image, template, info=None, matchImage=None):
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox image
    """

    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    if is_debugging and 'saver' in info:
        saver = info['saver']
        info['imageKeyPointNum'] = len(imageKeyPoint)
        info['templateKeyPointNum'] = len(templateKeyPoint)
        templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
        imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
        saver.saveImg(templateBlurred, 'template_key_points')
        saver.saveImg(imageBlurred, 'image_key_points')
        # cv2.imshow("template", templateBlurred)
        # cv2.imshow("image", imageBlurred)
        # cv2.waitKey(0)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]

    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    good2 = []
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # print(diff)
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]
    if len(good2) < 3:
        print('Not found')
        return template
    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    if is_debugging and 'saver' in info:
        saver = info['saver']
        matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
        saver.saveImg(matchImage, 'shift_match')
        info['averageDistance '] = averageDistance
        # cv2.imshow("matchImage", matchImage)
        # cv2.waitKey(0)
    else:
        matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)

    matchPointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good2])

    # for p1, p2 in matchPointMatrix:
    #     cv2.circle(image, (int(p1), int(p2)), 0, (255, 0, 0), thickness=50)
    #     print(p1, p2)
    # cv2.imshow("matchImage", image)

    minX = int(np.min(matchPointMatrix[:, 0]))
    maxX = int(np.max(matchPointMatrix[:, 0]))
    minY = int(np.min(matchPointMatrix[:, 1]))
    maxY = int(np.max(matchPointMatrix[:, 1]))

    return image[minY:maxY, minX:maxX]


def meterFinderBySIFT2(image, template, info=None, matchImage=None):
    """
    locate meter's bbox
    :param image: image
    :param info: info
    :return: bbox image
    """
    template = info["template"]

    # cv2.imshow("template", template)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    # startPointUp = (info["startPointUp"]["x"], info["startPointUp"]["y"])
    # endPointUp = (info["endPointUp"]["x"], info["endPointUp"]["y"])
    # centerPointUp = (info["centerPointUp"]["x"], info["centerPointUp"]["y"])

    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    if is_debugging and 'saver' in info:
        saver = info['saver']
        templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
        imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
        saver.saveImg(templateBlurred, 'template_key_points')
        saver.saveImg(imageBlurred, 'image_key_points')
        # cv2.imshow("template", templateBlurred)
        # cv2.imshow("image", imageBlurred)
        # cv2.waitKey(0)
    info['imageKeyPointNum'] = len(imageKeyPoint)
    info['templateKeyPointNum'] = len(templateKeyPoint)

    # for debug
    # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    # cv2.imshow("template", templateBlurred)
    # cv2.imshow("image", imageBlurred)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]

    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]
    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)
    if is_debugging and 'saver' in info:
        saver = info['saver']
        matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
        saver.saveImg(matchImage, 'shift_match')
        info['averageDistance '] = averageDistance
        # cv2.imshow("matchImage", matchImage)
        # cv2.waitKey(0)
    else:
        matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # not match
    if len(good2) < 3:
        print("not found!")
        return template

    # 寻找转换矩阵 M
    src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, _ = template.shape

    # 找出匹配到的图形的四个点和标定信息里的所有点
    pts = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [startPoint[0], startPoint[1]], [endPoint[0], endPoint[1]],
         [centerPoint[0], centerPoint[1]],
         # [startPointUp[0], startPointUp[1]],
         # [endPointUp[0], endPointUp[1]],
         # [centerPointUp[0], centerPointUp[1]]
         ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    if is_debugging and 'saver' in info:
        tl = dst[0][0]
        bl = dst[1][0]
        br = dst[2][0]
        tr = dst[3][0]
        image_copy = image.copy()
        cv2.line(image_copy, (tl[0], tl[1]), (tr[0], tr[1]), (0, 0, 255), 3)
        cv2.line(image_copy, (tr[0], tr[1]), (br[0], br[1]), (0, 0, 255), 3)
        cv2.line(image_copy, (br[0], br[1]), (bl[0], bl[1]), (0, 0, 255), 3)
        cv2.line(image_copy, (bl[0], bl[1]), (tl[0], tl[1]), (0, 0, 255), 3)
        saver.saveImg(image_copy, 'perspective_rect')

    # 校正图像
    angle = 0.0
    vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    if (vector[1] > 0):
        angle = math.acos(cos) * 180.0 / math.pi
    else:
        angle = -math.acos(cos) * 180.0 / math.pi
    # print(angle)

    change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
    src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 获得校正后的所需要的点
    newpoints = []
    for i in range(len(pts)):
        point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
                  int(round(newpoints[0][0])):int(round(newpoints[3][0]))]

    width = src_correct.shape[1]
    height = src_correct.shape[0]
    if width == 0 or height == 0:
        return template

    startPoint = (int(round(newpoints[4][0]) - newpoints[0][0]), int(round(newpoints[4][1]) - newpoints[0][1]))
    endPoint = (int(round(newpoints[5][0]) - newpoints[0][0]), int(round(newpoints[5][1]) - newpoints[0][1]))
    centerPoint = (int(round(newpoints[6][0]) - newpoints[0][0]), int(round(newpoints[6][1]) - newpoints[0][1]))

    def isOverflow(point, width, height):
        if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
            return True
        return False

    if isOverflow(startPoint, width, height) or isOverflow(endPoint, width, height) or isOverflow(centerPoint, width,
                                                                                                  height):
        print("overflow!")
        return template

    # startPointUp = (int(round(newpoints[7][0]) - newpoints[0][0]), int(round(newpoints[7][1]) - newpoints[0][1]))
    # endPointUp = (int(round(newpoints[8][0]) - newpoints[0][0]), int(round(newpoints[8][1]) - newpoints[0][1]))
    # centerPointUp = (int(round(newpoints[9][0]) - newpoints[0][0]), int(round(newpoints[9][1]) - newpoints[0][1]))
    info["startPoint"]["x"] = startPoint[0]
    info["startPoint"]["y"] = startPoint[1]
    info["centerPoint"]["x"] = centerPoint[0]
    info["centerPoint"]["y"] = centerPoint[1]
    info["endPoint"]["x"] = endPoint[0]
    info["endPoint"]["y"] = endPoint[1]

    return src_correct


class AngleFactory:
    """method for angle calculation"""

    @staticmethod
    def __calAngleBetweenTwoVector(vectorA, vectorB):
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        if abs(cosAngle - 1) <= 10e-4:
            return np.deg2rad(0)
        # notes that dot product calculates min angle between vectorA and vectorB only.
        angle = np.arccos(cosAngle)
        return angle

    @classmethod
    def calAngleBetweenTwoVector(cls, vectorA, vectorB):
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        angle = np.arccos(cosAngle)
        if abs(cosAngle - 1) <= 10e-4:
            return np.deg2rad(0)
        return angle

    @classmethod
    def calAngleClockwise(cls, startPoint, endPoint, centerPoint):
        """
        get clockwise angle formed by three point
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :return: clockwise angle
        """
        vectorA = startPoint - centerPoint
        vectorB = endPoint - centerPoint
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle
        # if counter-clockwise
        # if cross product(two-dim vector's cross product returns a integer only)
        # is negative ,means the normal vector is oriented down,vectorA is in the clockwise of vectorB
        # otherwise in counterclockwise.
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle
        return angle

    @classmethod
    def calAngleClockwiseByVector(cls, vectorA, vectorB):
        """
        get clockwise angle formed by two vector
        """
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle
        # if counter-clockwise
        # if cross product(two-dim vector's cross product returns a integer only)
        # is negative ,means the normal vector is oriented down,vectorA is in the clockwise of vectorB
        # otherwise in counterclockwise.
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        return angle

    @classmethod
    def calPointerValueByOuterPoint(cls, startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue):
        """
        get value of pointer meter
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param pointerPoint: pointer's outer point
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)
        angle = cls.calAngleClockwise(startPoint, pointerPoint, centerPoint)
        value = angle / angleRange * totalValue + startValue

        return value

    @classmethod
    def calPointerValueByPointerVector(cls, startPoint, endPoint, centerPoint, PointerVector, startValue, totalValue):
        """
        get value of pointer meter
        注意传入相对圆心的向量
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param PointerVector: pointer's vector
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = PointerVector

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue

        return value

    @classmethod
    def calPointerValueByPoint(cls, startPoint, endPoint, centerPoint, point, startValue, totalValue):
        """
        由三个点返回仪表值,区分@calPointerValueByPointerVector
        :param startPoint: 起点
        :param endPoint: 终点
        :param centerPoint:
        :param point:
        :param startValue:
        :param totalValue:
        :return:
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = point - centerPoint

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)
        if angle == np.deg2rad(0):
            return angle

        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue

        return value

    def findPointerFromHSVSpace(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5,
                                low_ptr_color=np.array([0, 0, 221]), up_ptr_color=np.array([180, 30, 255])):

        """
        从固定颜色的区域找指针,未完成
        :param low_ptr_color: 指针的hsv颜色空间的下界
        :param up_ptr_color:  指针的hsv颜色空间的上界
        :param radians_low:圆的搜索范围(弧度制表示)
        :param radians_high:圆的搜索范围(弧度制表示)
        :param src: 二值图
        :param center: 刻度盘的圆心
        :param radius: 圆的半径
        :param patch_degree:搜索梯度，默认每次一度
        :param ptr_resolution: 指针的粗细程度
        :return: 指针遮罩、直线与圆相交的点
        """

    pass


def scanPointer(meter, info):
    """
    find pointer of meter
    :param meter: meter matched template
    :param pts: a list including three numpy array, eg: [startPointer, endPointer, centerPointer]
    :param startVal: an integer of meter start value
    :param endVal: an integer of meter ending value
    :return: pointer reading number
    """
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    startVal = info["startValue"]
    endVal = info["totalValue"]
    if meter.shape[0] > 500:
        fixHeight = 300
        fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
        resizeCoffX = fixWidth / meter.shape[1]
        meter = cv2.resize(meter, (fixWidth, fixHeight))

        start = (start * resizeCoffX).astype(np.int32)
        end = (end * resizeCoffX).astype(np.int32)
        center = (center * resizeCoffX).astype(np.int32)

    radius = int(EuclideanDistance(start, center))

    src = cv2.GaussianBlur(meter, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)

    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    cv2.circle(mask, (center[0], center[1]), radius, (255, 255, 255), -1)
    thresh = cv2.bitwise_and(thresh, mask)
    cv2.circle(thresh, (center[0], center[1]), int(radius / 3), (0, 0, 0), -1)

    thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), 3)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8))

    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    startAngle = int(
        AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                       endPoint=start) * 180 / np.pi)
    endAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                                  endPoint=end) * 180 / np.pi)
    radius_range = radius * info['searchRadius']
    # print(startAngle, endAngle)
    if endAngle <= startAngle:
        endAngle += 360
    maxSum = 0
    outerPoint = start
    for angle in range(startAngle - 10, endAngle + 10):
        pts, outPt = getPoints(center, radius_range, angle)
        thisSum = 0
        showImg = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

        for pt in pts:
            cv2.circle(showImg, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            if pt[0] < thresh.shape[1] and pt[1] < thresh.shape[0] and thresh[pt[1], pt[0]] != 0:
                thisSum += 1

        # cv2.circle(showImg, (outPt[0], outPt[1]), 2, (255, 0, 0), -1)
        # cv2.imshow("img", showImg)
        # cv2.waitKey(1)
        if thisSum > maxSum:
            maxSum = thisSum
            outerPoint = outPt

    if start[0] == outerPoint[0] and start[1] == outerPoint[1]:
        degree = startVal
    elif end[0] == outerPoint[0] and end[1] == outerPoint[1]:
        degree = endVal
    else:
        if start.all() == end.all():
            end[0] -= 1
            end[1] -= 3
        degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, outerPoint, startVal, endVal)

    # small value to zero
    if degree < 0.05 * endVal:
        degree = startVal

    return degree, outerPoint


def EuclideanDistance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def getPoints(center, radious, angle):
    res = []
    farthestPointX = int(center[0] + radious * np.cos(angle / 180 * np.pi))
    farthestPointY = int(center[1] + radious * np.sin(angle / 180 * np.pi))

    delta_y = farthestPointX - center[0]
    delta_y = delta_y if delta_y != 0 else delta_y + 1
    k = (farthestPointY - center[1]) / delta_y
    b = center[1] - k * center[0]

    for x in range(min(farthestPointX, center[0]), max(farthestPointX, center[0])):
        for y in range(min(farthestPointY, center[1]), max(farthestPointY, center[1])):
            if k * x + b - 2 <= y <= k * x + b + 2:
                res.append([x, y])

    return res, [farthestPointX, farthestPointY]


def findPointerFromBinarySpace(src, center, radius, radians_low=0, radians_high=2 * np.pi, patch_degree=1.0,
                               ptr_resolution=5, clean_ration=0.16, avg_len=None):
    """
    接收一张预处理过的二值图（默认较完整保留了指针信息），指针的轮廓应为白色，
    从通过圆心水平线与圆的左交点开始，连接圆心顺时针建立直线遮罩，取出遮罩范围下的区域,
    计算对应区域灰度和，灰度和最大的区域即为指针所在的位置。直线遮罩的粗细程度、搜索的梯度决定了算法侦测指针的细粒度。该算法适合搜索指针形状
    为直线的仪表盘。
    :param radians_low:圆的搜索起点(弧度制表示),默认从0 处开始
    :param radians_high:圆的搜索终点(弧度制表示),默认从在2 * np.pi 处结束
    :param src: 二值图
    :param center: 刻度盘的圆心
    :param radius: 圆的半径
    :param patch_degree:搜索梯度，默认每次一度
    :param ptr_resolution: 指针的粗细程度(分辨率)
    :param clean_ration: 用圆遮罩清除圆心区域的元素,半径为 shape[0] * clean_ration
    :return: 被认为是指针区域的白色遮罩(黑色背景)、指针轮廓直线与圆相交的点
    """
    _shape = src.shape
    # img = src.copy()
    # img = cleanShortPart(img, center, _shape, clean_ration)
    # 弧度转化为角度值
    img = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)
    low = math.degrees(radians_low)
    high = math.degrees(radians_high)
    mask_info = []
    iteration = np.abs(int((high - low) / patch_degree))
    avg_len *= 3
    for i in range(iteration):
        # 建立一个大小跟输入一致的全黑图像
        # 每次旋转patch_degree度，取圆上一点
        if radians_low < 0:
            # 为了适应人读表的感观,将扫描方向转换为顺时针,起点初始为传入的弧度角
            theta = np.pi - np.radians(i * patch_degree) - radians_low
        else:
            theta = np.pi - np.radians(i * patch_degree) + radians_low

        pointer_mask, point = drawLineMask(_shape, theta, center, ptr_resolution, radius)
        # 去除遮罩对应的小区域
        and_img = cv2.bitwise_and(pointer_mask, img)
        img, white_contours, h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        not_zero_intensity = cv2.countNonZero(and_img)
        if avg_len is not None and not_zero_intensity < avg_len:
            continue
        mask_info.append((not_zero_intensity, theta))
    if not len(mask_info):
        return None, -1, [-1, -1]
    # 按灰度和从大到小排列
    mask_info = sorted(mask_info, key=lambda m: m[0], reverse=True)
    best_theta = mask_info[0][1] % 360
    # 得到灰度和最大的那个直线遮罩,和直线与圆相交的点
    pointer_mask, point = drawLineMask(_shape, best_theta, center, ptr_resolution, radius)
    best_theta = 180 - best_theta * 180 / np.pi
    if best_theta < 0:
        best_theta = 360 - best_theta
    return pointer_mask, best_theta, point


def cleanShortPart(src, center, shape, clean_ration):
    """
    根据标定信息清楚一些有干扰性的区域
    :param src:
    :param info:
    :param shape:
    :return:
    """
    circle_mask = cv2.bitwise_not(np.zeros(shape, dtype=np.uint8))
    cv2.circle(circle_mask, (np.int64(center[0]), np.int64(center[1])), np.int64(shape[0] * clean_ration),
               color=(0, 0, 255),
               thickness=cv2.FILLED)
    return cv2.bitwise_and(src, circle_mask)


def drawLineMask(_shape, theta, center, ptr_resolution, radius):
    """
    画一个长为radius，白色的直线，产生一个背景全黑的白色直线遮罩
    :param _shape:
    :param theta:
    :param center:
    :param ptr_resolution:
    :param radius:
    :return:
    """
    pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
    y1 = int(center[1] - np.sin(theta) * radius)
    x1 = int(center[0] + np.cos(theta) * radius)
    cv2.line(pointer_mask, (center[0], center[1]), (x1, y1), 255, ptr_resolution)
    return pointer_mask, (x1, y1)


def detectHoughLine(meter, cannyThresholds, houghParam):
    '''
    detect pointer of meter
    :param meter:
    :param cannyThresholds: [threshold1, threshold2], parameters of function Canny
    :param houghParam:
    :return:
    '''
    img = cv2.GaussianBlur(meter, (3, 3), 0)  # cv2.imshow("GaussianBlur ", img)
    edges = cv2.Canny(img, cannyThresholds[0], cannyThresholds[1], apertureSize=3)
    # cv2.imshow("canny", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, houghParam)  # 这里对最后一个参数使用了经验型的值
    if lines is None:
        return None
    height, width, _ = img.shape
    pointer = []
    rho, theta = lines[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # print("x0, y0:", x0, y0)
    # print("width, height ", width, height)
    xcenter = int(width / 2)
    ycenter = int(height / 2)
    if xcenter < x0 or (xcenter == x0 and ycenter > y0):
        x1 = xcenter
        x2 = x0
        y1 = ycenter
        y2 = y0
    else:
        x1 = x0
        x2 = xcenter
        y1 = y0
        y2 = ycenter
    cv2.line(meter, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("HoughLine", meter)
    cv2.waitKey(0)
    pointer.append([x2 - x1, y2 - y1])
    pointer = np.array(pointer[0])
    return pointer


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        sigma = 10
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype(np.uint8)
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out.astype(np.uint8)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.astype(np.uint8)
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy.astype(np.uint8)


def round_res(res, precision, rounding=None):
    """
    rounding res , default format is ROUND_HALF_EVEN
    :param res:
    :param precision:
    :param rounding:
    :return:
    """
    format = '0.'
    for i in range(precision):
        format = format + '0'
    decimal = Decimal(res)
    if rounding is None:
        rounding = ROUND_HALF_EVEN
    quantize = decimal.quantize(Decimal(format), rounding=rounding)
    return float(quantize)


def enhance(src, meter_id, out_main_dir):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    save_dir = out_main_dir + os.path.sep + meter_id + '_0_gray' + '.png'
    cv2.imwrite(save_dir, gray)
    normalize = np.zeros(gray.shape)
    cv2.normalize(gray, normalize, 0, 1, cv2.NORM_MINMAX)
    # plot.subImage(src=normalize, index=plot.next_idx(), title='Gray', cmap='gray')
    # plot.subImage(src=gray, index=plot.next_idx(), title='Gray', cmap='gray')
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    laplace = cv2.filter2D(gray, cv2.CV_8UC1, laplace_kernel)
    # laplace = cv2.Laplacian(gray, cv2.CV_8UC1, ksize=3)
    save_dir = out_main_dir + os.path.sep + meter_id + '_1_enhance_laplace' + '.png'
    cv2.imwrite(save_dir, laplace)
    # abs_labplace = cv2.convertScaleAbs(laplace)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Laplace', cmap='gray')
    # cv2.normalize(abs_labplace, abs_labplace)
    overlay = cv2.add(gray, laplace)
    save_dir = out_main_dir + os.path.sep + meter_id + '_2_overlay' + '.png'
    cv2.imwrite(save_dir, overlay)
    # plot.subImage(src=overlay, index=plot.next_idx(), title='Overlay by Gray + Laplace', cmap='gray')
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3))
    grad = cv2.convertScaleAbs(cv2.add(sobel_x, sobel_y))
    save_dir = out_main_dir + os.path.sep + meter_id + '_3_grad_' + '.png'
    cv2.imwrite(save_dir, grad)
    # cv2.normalize(grad, grad, 1.0, 0.0, cv2.NORM_MINMAX)
    # plot.subImage(src=grad, index=plot.next_idx(), title='Grad', cmap='gray')
    # media_blur = cv2.medianBlur(grad, 5)
    save_dir = out_main_dir + os.path.sep + meter_id + '_4_grad_blurred' + '.png'
    mean_blur = cv2.blur(gray, (5, 5))
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(mean_blur, cv2.CV_8U, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(mean_blur, cv2.CV_8U, 0, 1, ksize=3))
    blur_grad = cv2.convertScaleAbs(cv2.add(sobel_x, sobel_y))
    cv2.imwrite(save_dir, blur_grad)
    # cv2.normalize(grad, grad, 1.0, 0.0, cv2.NORM_MINMAX)
    # plot.subImage(src=grad, index=plot.next_idx(), title='Grad', cmap='gray')
    # media_blur = cv2.medianBlur(grad, 5)
    # media_blur = np.float32(media_blur) * (1 / 255)
    # overlay = np.float32(overlay) * (1 / 255)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Media Blurred', cmap='gray')
    mask = cv2.bitwise_and(blur_grad, overlay)
    # mask = (mask > 1) * 255
    # np.where(mask >= 1, 255, 0)
    #  print(mask)
    save_dir = out_main_dir + os.path.sep + meter_id + '_4_sharp_mask' + '.png'
    cv2.imwrite(save_dir, mask)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Mask', cmap='gray')
    enhance = cv2.add(gray, mask)
    save_dir = out_main_dir + os.path.sep + meter_id + '_5_enhanced' + '.png'
    cv2.imwrite(save_dir, enhance)
    # plot.subImage(src=enhance, index=plot.next_idx(), title='Enhance', cmap='gray')
    return cv2.cvtColor(enhance, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_dir, media_blur)
    # media_blur = np.float32(media_blur) * (1 / 255)
    # overlay = np.float32(overlay) * (1 / 255)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Media Blurred', cmap='gray')
    mask = cv2.bitwise_and(media_blur, overlay)
    # mask = (mask > 1) * 255
    # np.where(mask >= 1, 255, 0)
    #  print(mask)
    save_dir = out_main_dir + os.path.sep + meter_id + '_4_sharp_mask' + '.png'
    cv2.imwrite(save_dir, mask)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Mask', cmap='gray')
    enhance = cv2.add(gray, mask)
    save_dir = out_main_dir + os.path.sep + meter_id + '_5_enhanced' + '.png'
    cv2.imwrite(save_dir, enhance)
    # plot.subImage(src=enhance, index=plot.next_idx(), title='Enhance', cmap='gray')
    return cv2.cvtColor(enhance, cv2.COLOR_GRAY2BGR)
