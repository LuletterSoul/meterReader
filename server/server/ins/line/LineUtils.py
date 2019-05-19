import numpy as np


def inSegment(p, line, line2):
    """
    检查某交点是否在线段line上（不含line的端点），在求交点时已经确认两条直线不平行
    所以，对于竖直的line，line2不可能竖直，却有可能水平，所以检查p是否在line2上，只能检查x值即p[0]
    """
    if line[0][0] == line[1][0]:  # 如果line竖直
        if min(line[0][1], line[1][1]) < p[1] < max(line[0][1], line[1][1]):
            # if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
            if min(line2[0][0], line2[1][0]) <= p[0] <= max(line2[0][0], line2[1][0]):
                return True
    elif line[0][1] == line[1][1]:  # 如果line水平
        if min(line[0][0], line[1][0]) < p[0] < max(line[0][0], line[1][0]):
            # if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
            if min(line2[0][1], line2[1][1]) <= p[1] <= max(line2[0][1], line2[1][1]):
                return True
    else:
        if min(line[0][0], line[1][0]) < p[0] < max(line[0][0], line[1][0]):
            # line为斜线时，line2有可能竖直也有可能水平，所以对x和y都需要检查
            if min(line2[0][1], line2[1][1]) <= p[1] <= max(line2[0][1], line2[1][1]) and min(
                    line2[0][0], line2[1][0]) <= p[0] <= max(line2[0][0], line2[1][0]):
                return True
    return False


def getLinePara(line):
    """简化交点计算公式"""
    a = line[0][1] - line[1][1]
    b = line[1][0] - line[0][0]
    c = line[0][0] * line[1][1] - line[1][0] * line[0][1]
    return a, b, c


def getCrossPoint(line1, line2, shape):
    """计算交点坐标，此函数求的是line1中被line2所切割而得到的点，不含端点"""
    a1, b1, c1 = getLinePara(line1)
    a2, b2, c2 = getLinePara(line2)
    d = a1 * b2 - a2 * b1
    p = np.array([0, 0])
    if d == 0:
        return [-1, -1]
    else:
        p[0] = round((b1 * c2 - b2 * c1) * 1.0 / d, 2)  # 工作中需要处理有效位数，实际可以去掉round()
        p[1] = round((c1 * a2 - c2 * a1) * 1.0 / d, 2)
        if p[0] > shape[0] or p[1] > shape[1] or p[0] < 0 or p[1] < 0:
            return [-1, -1]
    return p


def getDistPtToLine(pt, pta, ptb):
    a = pta[1] - ptb[1]
    b = ptb[0] - pta[0]
    c = pta[0] * ptb[1] - pta[1] * ptb[0]
    return np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a * a + b * b)
