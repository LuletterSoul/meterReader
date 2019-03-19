from Common import *
import json
import util.PlotUtil as plot
import imutils
import LineSegmentFilter as LSF

plot_index = 0


def normalPressure(image, info):
    '''
    :param image: ROI image
    :param info: information for this meter
    :return: value
    '''
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    meter = meterFinderByTemplate(image, info["template"])
    result1 = scanPointer(meter, [start, end, center], info["startValue"], info["totalValue"])
    result2 = readPressure(image, info)
    return [result1, result2]


def readPressure(image, info):
    src = meterFinderByTemplate(image, info["template"])
    plot.subImage(src=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Original Image')
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    thresh = gray.copy()
    cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV, thresh)
    # image thinning
    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # find contours
    img, contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # filter small contours.
    # contours = [c for c in contours if len(c) > contours_thresh]

    # draw contours
    filtered_thresh = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(filtered_thresh, contours, -1, (255, 0, 0), thickness=cv2.FILLED)
    thresh = filtered_thresh
    # plot.subImage(src=filtered_thresh, index=plot.next_idx(), title='Filtered Threshold', cmap='gray')
    # load meter calibration form configuration
    start_ptr = info['startPoint']
    end_ptr = info['endPoint']
    start_ptr = cvtPtrDic2D(start_ptr)
    end_ptr = cvtPtrDic2D(end_ptr)
    center = info['centerPoint']
    center = cvtPtrDic2D(center)
    # 起点和始点连接，分别求一次半径,并得到平均值
    radius = calAvgRadius(center, end_ptr, start_ptr)
    hlt = np.array([center[0] - radius, center[1]])  # 通过圆心的水平线与圆的左交点
    # 计算起点向量、终点向量与过圆心的左水平线的夹角
    start_radians = AngleFactory.calAngleClockwise(start_ptr, hlt, center)
    # 以过圆心的左水平线为扫描起点
    if start_radians < np.pi:
        # 在水平线以下,标记为负角
        start_radians = -start_radians
    end_radians = AngleFactory.calAngleClockwise(hlt, end_ptr, center)
    ptr_resolution = 5
    clean_ration = 0
    # 从特定范围搜索指针
    pointer_mask, theta, line_ptr = findPointerFromBinarySpace(thresh, center, radius, start_radians,
                                                               end_radians,
                                                               patch_degree=0.5,
                                                               ptr_resolution=ptr_resolution, clean_ration=clean_ration)
    line_ptr = cv2PtrTuple2D(line_ptr)
    plot.subImage(src=cv2.bitwise_or(thresh, pointer_mask), index=plot.next_idx(), title='pointer', cmap='gray')
    cv2.line(src, (start_ptr[0], start_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.line(src, (end_ptr[0], end_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.circle(src, (start_ptr[0], start_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(src, (end_ptr[0], end_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(src, (center[0], center[1]), 2, (0, 0, 255), -1)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Calibration Info')
    start_value = info['startValue']
    total = info['totalValue']
    value = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr,
                                                centerPoint=center,
                                                point=line_ptr, startValue=start_value,
                                                totalValue=total)
    return value


def calAvgRadius(center, end_ptr, start_ptr):
    radius_1 = np.sqrt(np.power(start_ptr[0] - center[0], 2) + np.power(start_ptr[1] - center[1], 2))
    radius_2 = np.sqrt(np.power(end_ptr[0] - center[0], 2) + np.power(end_ptr[1] - center[1], 2))
    radius = np.int64((radius_1 + radius_2) / 2)
    return radius


def cvtPtrDic2D(dic_ptr):
    """
    point.x,point.y转numpy数组
    :param dic_ptr:
    :return:
    """
    if dic_ptr['x'] and dic_ptr['y'] is not None:
        dic_ptr = np.array([dic_ptr['x'], dic_ptr['y']])
    else:
        return np.array([0, 0])
    return dic_ptr


def cv2PtrTuple2D(tuple):
    """
    tuple 转numpy 数组
    :param tuple:
    :return:
    """
    if tuple[0] and tuple[1] is not None:
        tuple = np.array([tuple[0], tuple[1]])
    else:
        return np.array([0, 0])
    return tuple


def readPressureValueFromDir(meter_id, img_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    assert info is not None
    info["template"] = cv2.imread("template/" + meter_id + ".jpg")
    return readPressureValueFromImg(img, info)


def readPressureValueFromImg(img, info):
    if img is None:
        raise Exception("Failed to resolve the empty image.")
    return normalPressure(img, info)


def initExtractScaleLine(meter_id, img_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    assert info is not None
    info["template"] = cv2.imread("template/" + meter_id + ".jpg")
    extractScaleLine(img, info)


def extractScaleLine(image, info):
    src = meterFinderByTemplate(image, info["template"])
    # src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0)
    # src = cv2.fastNlMeansDenoisingColored(src, h=7, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)
    gray_test = gray.copy()
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Fast denosing')
    retval, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # gray_test, covex_mask = LSF.filterContex(gray_test)
    # Tresh + Gray Convex Masx
    # thresh_pre = cv2.bitwise_and(thresh, covex_mask)
    # plot.subImage(src=thresh_pre, index=plot.next_idx(), title='lhsh conver', cmap='gray')
    thresh_convex, tm = LSF.filterContex(thresh)
    plot.subImage(src=thresh_convex, index=plot.next_idx(), title='Thresh conver', cmap='gray')
    plot.subImage(src=tm, index=plot.next_idx(), title='Thresh tm', cmap='gray')
    # plot.subImage(src=thresh, index=plot.next_idx(), title='Thresh_OTSU', cmap='gray')
    canny = cv2.Canny(src, 75, 75 * 2)
    auto_canny = imutils.auto_canny(gray)
    dilate_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
    erode_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
    # fill scale line with white pixels
    canny = cv2.dilate(canny, dilate_kernel)
    canny = cv2.erode(canny, erode_kernel)
    auto_canny = cv2.dilate(auto_canny, dilate_kernel)
    auto_canny = cv2.erode(auto_canny, erode_kernel)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Auto Canny', cmap='gray')
    auto_canny = LSF.clean(auto_canny)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Auto Canny + Cask', cmap='gray')
    auto_canny = cv2.ximgproc.thinning(auto_canny, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    detector = cv2.createLineSegmentDetector()
    _lines, width, prec, nfa = detector.detect(auto_canny)
    line_src = np.zeros(src.shape, dtype=np.uint8)
    detector.drawSegments(line_src, _lines)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Thining', cmap='gray')
    plot.subImage(src=line_src, index=plot.next_idx(), title='Line Detector')

    # Auto Canny + Mask


if __name__ == '__main__':
    # res1 = readPressureValueFromDir('lxd1_4', 'image/lxd1.jpg', 'config/lxd1_4.json')
    # res2 = readPressureValueFromDir('szk2_1', 'image/szk2.jpg', 'config/szk2_1.json')
    # res3 = readPressureValueFromDir('szk1_5', 'image/szk1.jpg', 'config/szk1_5.json')
    # res4 = readPressureValueFromDir('wn1_5', 'image/wn1.jpg', 'config/wn1_5.json')
    # res5 = readPressureValueFromDir('xyy3_1', 'image/xyy3.jpg', 'config/xyy3_1.json')
    # res6 = readPressureValueFromDir('pressure2_1', 'image/pressure2_1.jpg', 'config/pressure2_1.json')
    # res7 = readPressureValueFromDir('lxd1_2', 'image/lxd1.jpg', 'config/lxd1_2.json')
    initExtractScaleLine('pressure2_1', 'image/pressure2_1.jpg', 'config/pressure2_1.json')
    initExtractScaleLine('lxd1_2', 'image/lxd1.jpg', 'config/lxd1_2.json')
    plot.show()
    # print(res7)
