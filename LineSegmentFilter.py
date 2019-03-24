from Common import *
from collections import defaultdict
import imutils
import skimage.measure as skm
from LineUtils import getCrossPoint, getDistPtToLine
import util.PlotUtil as plot

thresh1 = 20
thresh2 = 25
thresh3 = 10


def calLineVector(line_a):
    return np.array([line_a[2] - line_a[0], line_a[3] - line_a[1]])


def calCenter(line_a):
    return np.array([(line_a[0] + line_a[2]) / 2, (line_a[1] + line_a[3]) / 2])


thresh = 50
N = 11


def findSquares(img):
    shape = img.shape
    pyr = cv2.pyrDown(img)
    timg = cv2.pyrUp(pyr)
    gray0 = np.zeros(img.shape, dtype=np.uint8)
    gray = np.zeros(img.shape, dtype=np.uint8)
    squares = []
    gray0 = cv2.split(timg)
    for c in range(0, len(gray0)):
        for l in range(0, N):
            if l == 0:
                gray = cv2.Canny(image=gray0[c], threshold1=0, threshold2=5,
                                 apertureSize=5)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))
                gray = cv2.dilate(gray, kernel)
            else:
                _, gray = cv2.threshold(gray0[c], int((l + 1) * 255 / N), 255, cv2.THRESH_BINARY_INV)
            gray, contours, hierarcy = cv2.findContours(np.array(gray, dtype=np.uint8), method=cv2.RETR_LIST,
                                                        mode=cv2.CHAIN_APPROX_NONE)
            for k in range(0, len(contours)):
                approx = cv2.approxPolyDP(contours[k], cv2.arcLength(contours[k], True) * 0.02, True)
                if len(approx) == 4 and np.abs(cv2.contourArea(approx) > 1000) and cv2.isContourConvex(approx):
                    # cv2.drawContours(img, contours, k, (0, 255, 0))
                    # cv2.polylines(img, approx, True, (255, 0, 0), 3, cv2.FILLED)
                    maxCosine = 0
                    for j in range(2, 5):
                        vectorA = (approx[j % 4] - approx[j - 1])[0]
                        vectorB = (approx[j - 2] - approx[j - 1])[0]
                        cosine = np.cos(AngleFactory.calAngleBetweenTwoVector(vectorA, vectorB))
                        maxCosine = max(cosine, maxCosine)
                    if maxCosine < 0.3:
                        squares.append(approx)
        return squares


def filterContex(src):
    pyr = cv2.pyrDown(src)
    timg = cv2.pyrUp(pyr)
    timg = cleanExternalContours(timg)
    mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
    for l in range(0, N):
        if l == 0:
            gray = cv2.Canny(image=timg, threshold1=0, threshold2=5,
                             apertureSize=5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))
            gray = cv2.dilate(gray, kernel)
        else:
            _, gray = cv2.threshold(timg, int((l + 1) * 255 / N), 255, cv2.THRESH_BINARY_INV)
        gray, contours, hierarcy = cv2.findContours(gray, method=cv2.RETR_TREE,
                                                    mode=cv2.CHAIN_APPROX_NONE)
        lens = []
        for i, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
            cv2.boundingRect(approx)
            lens.append(len(approx))
            if len(approx) >= 10:
                cv2.drawContours(mask, [contour], -1, -1, cv2.FILLED)
            # print("Max", max(lens))
            # print("Min", min(lens))
            # print("Avg", all_approx_size / len(contours))
    src = cv2.bitwise_and(src, src, mask=mask)
    return src, mask


def cleanExternalContours(src, output=None):
    _, external_contours, _ = cv2.findContours(src, method=cv2.RETR_LIST, mode=cv2.CHAIN_APPROX_NONE)
    clean_external_mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
    max_area = cv2.contourArea(max(external_contours, key=cv2.contourArea))
    min_area = cv2.contourArea(min(external_contours, key=cv2.contourArea))
    high = (max_area - min_area) * 0.6
    contours = [c for c in external_contours if cv2.contourArea(c) - min_area >= high]
    sorted(external_contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        cv2.drawContours(clean_external_mask, [c], -1, -1, 3)
    # cv2.imshow("Mask", clean_external_mask)
    # cv2.waitKey(0)
    output = cv2.bitwise_and(src, src, mask=clean_external_mask)
    return output


def cleanNotInterestedFeature(src, area_thresh=500, approx_thresh=8, new_src=None):
    pyr = cv2.pyrDown(src)
    timg = cv2.pyrUp(pyr)
    new_src = np.zeros(src.shape[:2], dtype=np.uint8)
    gray, contours, hierarcy = cv2.findContours(timg, method=cv2.RETR_LIST,
                                                mode=cv2.CHAIN_APPROX_NONE)
    max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
    min_area = cv2.contourArea(min(contours, key=cv2.contourArea))
    high = (max_area - min_area) * 0.7
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
        if len(approx) <= approx_thresh and cv2.contourArea(contour) < area_thresh:
            cv2.drawContours(new_src, [contour], -1, 255, cv2.FILLED)
    return new_src


def cleanNotInterestedFeatureByProps(src, area_thresh=500, approx_thresh=8, rect_ration_thresh=30, new_src=None):
    # debug_src = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    # sub = 3
    # # src = src[sub:src.shape[1] - sub][sub:src.shape[0] - sub]
    # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    # print(labels)
    # coords = defaultdict(list)
    # for i in range(src.shape[1]):
    #     for j in range(src.shape[0]):
    #         coords[labels[0][i, j]].append(np.array(i, j))
    # print(centroids)

    # coords = [pt for pt in labels if pt]
    # print(len(centroids))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # regions = skm.regionprops(src)
    # print(len(regions))
    # pts = [region.coords for region in regions if inThresholdRange(region, area_thresh, 10)]
    new_src = np.zeros(src.shape[:2], dtype=np.uint8)
    # contour_src = np.zeros(src.shape[:2], dtype=np.uint8)
    src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    gray, contours, hierarcy = cv2.findContours(src, method=cv2.RETR_LIST,
                                                mode=cv2.CHAIN_APPROX_NONE)
    for i, contour in enumerate(contours):
        if inThresholdRange(contour, src.shape, approx_thresh, area_thresh, rect_ration_thresh):
            cv2.drawContours(new_src, [contour], -1, 255, cv2.FILLED)
    return new_src


def inThresholdRange(pts, shape, approx_thresh, area_thresh, rect_ration_thresh):
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
    rect = cv2.minAreaRect(pts)
    rect_width = np.int(rect[1][0])
    rect_height = np.int(rect[1][1])
    if rect_width == 0 or rect_height == 0:
        return False
    ratio = min(rect_width, rect_height) / max(rect_width, rect_height) * 100
    ration_in_range = ratio < rect_ration_thresh
    area_in_range = cv2.contourArea(pts) < area_thresh
    approx_in_range = len(approx) < approx_thresh
    len_in_range = cv2.arcLength(pts, False) < max(shape[0], shape[1]) * 0.5
    return ration_in_range and area_in_range and len_in_range


def cleanPoly(src, poly):
    mask = np.ones(src.shape, dtype=np.uint8) * 255
    for p in poly:
        cv2.drawContours(mask, [p], -1, (0, 0, 0), 0)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
    src = cv2.bitwise_and(src, mask)
    return src


def drawSquares(img, squares):
    for k in range(0, len(squares)):
        cv2.polylines(img, squares[k], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Squares", img)
    cv2.waitKey(0)


def filter(_lines, shape, _filtered_lines=None):
    _filtered_lines = []
    line_descriptors = []
    level_line_vector = np.array([0, 1])
    for line in _lines:
        line = line[0]
        center = calCenter(line)
        start_ptr = np.array([line[0], line[1]])
        end_ptr = np.array([line[2], line[3]])
        line_len = EuclideanDistance(start_ptr, end_ptr)  # distance between start point and end point
        if line_len > 30:
            continue
        line_vector = calLineVector(line)  # line vector
        line_descriptors.append(
            np.array([start_ptr, end_ptr, line_vector, center, line_len]))  # compose line descriptors

    match_pairs = []
    line_set_len = len(line_descriptors)
    vis = np.zeros(shape=(line_set_len, 1), dtype=np.int32)
    for i in range(0, line_set_len):
        if vis[i] == 1:
            continue
        for j in range(0, line_set_len):
            if i == j or vis[j] == 1:
                continue
            # t = cv2.getTickCount()
            dis_center = EuclideanDistance(line_descriptors[i][3], line_descriptors[j][3])
            hor_angle1 = AngleFactory.calAngleClockwiseByVector(level_line_vector, line_descriptors[i][2])
            hor_angle2 = AngleFactory.calAngleClockwiseByVector(level_line_vector, line_descriptors[j][2])
            if hor_angle1 >= 1.5 * np.pi:
                hor_angle1 = hor_angle1 - np.pi
            if hor_angle2 >= 1.5 * np.pi:
                hor_angle2 = hor_angle2 - np.pi
            diff_angle = np.rad2deg(np.abs(hor_angle2 - hor_angle1))
            diff_len = np.abs(line_descriptors[i][4] - line_descriptors[j][4])
            # n = (cv2.getTickCount() - t) / cv2.getTickFrequency()
            if dis_center < thresh1 or diff_angle < thresh3:
                line1 = [line_descriptors[i][0], line_descriptors[i][1]]
                line2 = [line_descriptors[j][0], line_descriptors[j][1]]
                match_pairs.append([line1, line2])
                _filtered_lines.append(_lines[i])
                _filtered_lines.append(_lines[j])
                vis[i] = 1
                vis[j] = 1
                break
    _filtered_lines, avg_center = filterMatchedLine(match_pairs, _filtered_lines, shape, )
    avg_center = np.int32(avg_center)
    # cv2.circle(draw_src, (avg_center[0], avg_center[1]), 4, (0, 255, 0), cv2.FILLED)
    # line_deector.drawSegments(draw_src, np.array(_filtered_lines))
    # line_detector.drawSegments(draw_src, _lines)
    return np.array(_filtered_lines), avg_center


def filterMatchedLine(pairs, matched_lines, shape, filtered_lines=None):
    left_lines = [l[0] for l in pairs]
    right_lines = [l[1] for l in pairs]
    left_cross_pts = getCrossPointerSet(left_lines, shape)
    right_cross_pts = getCrossPointerSet(right_lines, shape)
    if not len(left_cross_pts) or not len(right_cross_pts):
        raise Exception("Cross pointer set couldn't be empty.")
    avg_center = (np.mean(left_cross_pts, axis=0) + np.mean(right_cross_pts, axis=0)) / 2
    filtered_lines = [l for l in matched_lines if
                      getDistPtToLine(avg_center, [l[0][0], l[0][1]], [l[0][2], l[0][3]]) < min(shape[0],
                                                                                                shape[1]) * 0.2]
    return filtered_lines, avg_center


def getCrossPointerSet(lines, shape, cross_pts=None):
    cross_pts = []
    len_line = len(lines)
    for i in range(0, len_line):
        for j in range(0, len_line):
            if i == j:
                continue
            pt = getCrossPoint(lines[i], lines[j], shape)
            if pt[0] == -1 and pt[1] == -1:
                continue
            cross_pts.append(pt)
    return np.array(cross_pts)


if __name__ == '__main__':
    names = ['data/pic1.png', 'data/pic2.png', 'data/pic3.png', 'data/pic4.png', 'data/pic5.png', 'data/pic6.png']
    for i in range(0, len(names)):
        print("Image name is :", names[i])
        img = cv2.imread(names[i])
        if img is None:
            print("Couldn't load ", names[i])
        squares = findSquares(img)
        img = cleanPoly(img, squares)
        # drawSquares(img, squares)
        cv2.imshow("Img ", img)
        cv2.waitKey(0)
