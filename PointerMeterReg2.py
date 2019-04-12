from Common import *
from DebugSwitcher import is_plot, is_save
import json
import util.PlotUtil as plot
from util import RasancFitCircle as rasan
from util.StoreUtil import DataSaver
import random
import os
import time
from DeHaze import deHaze
import imutils
import LineSegmentFilter as LSF
import LineUtils as LU

saver = DataSaver()


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


def read(image, info):
    if info['matchTemplateType'] == 1:
        src = meterFinderBySIFT(image, info["template"], info)
    else:
        src = meterFinderByTemplate(image, info["template"])
    denoised = cv2.fastNlMeansDenoisingColored(src)
    # enhance(src)
    saver.saveImg(src, 'image_by_shift')
    saver.saveImg(denoised, 'denoising')
    src = denoised
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Original Image')
    if info['enableGaussianBlur']:
        src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    copy_src = src.copy()
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
    center = [-1, -1]
    radius = 0
    start_ptr = [-1, -1]
    end_ptr = [-1, -1]
    avg_len = 0
    start_value = info['startValue']
    total = info['totalValue']
    if info['enableFitting']:
        model, start_ptr, end_ptr, avg_len = getPointerInstrumentModel(src, info)
        if model[0] == -1:
            return -1
        center = np.array([model[0], model[1]])
        radius = model[2]
    else:
        center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
        start_ptr = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
        end_ptr = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    cv2.line(copy_src, (start_ptr[0], start_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.line(copy_src, (end_ptr[0], end_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.circle(copy_src, (start_ptr[0], start_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(copy_src, (end_ptr[0], end_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(copy_src, (center[0], center[1]), 2, (0, 0, 255), -1)
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
    pt_reg_alg_type = info['ptRegAlgType']
    if pt_reg_alg_type == 0:
        # 从特定范围搜索指针
        pointer_mask, theta, line_ptr = findPointerFromBinarySpace(thresh, center, radius * info['searchRadius'],
                                                                   start_radians,
                                                                   end_radians,
                                                                   patch_degree=0.5,
                                                                   ptr_resolution=ptr_resolution,
                                                                   clean_ration=clean_ration,
                                                                   avg_len=avg_len)
        if pointer_mask is not None:
            pm = cv2.bitwise_or(thresh, pointer_mask)
            plot.subImage(src=pm, index=plot.next_idx(), title='pointer', cmap='gray')
            saver.saveImg(pm, 'pointer_mast')
    elif pt_reg_alg_type == 1:
        # value, line_ptr = scanPointer(src, [start_ptr, end_ptr, center], start_value, total)
        value, line_ptr = scanPointer(src, info)
    cv2.circle(copy_src, (line_ptr[0], line_ptr[1]), 5, (0, 255, 0), cv2.FILLED)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Calibration Info')
    saver.saveImg(copy_src, 'model_center_range')
    if line_ptr[0] == -1:
        return 0
    line_ptr = cv2PtrTuple2D(line_ptr)
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


def load(meter_id, img_dir, config):
    global saver
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    if info['type'] != 'normalPressure':
        return -1
    assert info is not None
    saver = DataSaver('data/', meter_id)
    saver.saveImg(img, 'src')
    print("Img: ", meter_id)
    info["template"] = cv2.imread("template/" + meter_id + ".jpg")
    info["saver"] = saver
    start = time.time()
    res = read(img, info)
    end = time.time()
    consumption = round(end - start, 3)
    print("Reading Value: ", res)
    if 'realValue' in info:
        print("Real Value:", info['realValue'])
        real_value = info['realValue']
        # calculate absolute error
        abs_error = res - real_value / info['totalValue'] - info['startValue'] * 100
        abs_error = round(abs_error, 3)
        if res > 0:
            ration = (res - real_value) / res * 100
        info['res'] = res
        info['absError'] = str(abs_error) + ' %'
        info['consumption'] = consumption
        print('Absolute error :{} %'.format(abs_error))
        print('Time consumption:{}'.format(consumption))
        saver.saveConfig(info)
    print()
    return res


def readPressureValueFromImg(img, info):
    if img is None:
        raise Exception("Failed to resolve the empty image.")
    return normalPressure(img, info)


def init(meter_id, img_dir, config):
    img, info = load(config, img_dir, meter_id)
    getPointerInstrumentModel(img, info)


def getPointerInstrumentModel(img, info):
    """
    this function will get a fitting instrument model from a ideal image
    :param img:  input image
    :param info: contains all adjustable and configurable parameters,related to the intelligent instrument reading
                 algorithm
    :return:    geometric model of pointer instrument like circle , ellipse .
                two key points to settle reading range
    """
    model, start_pt, end_pt, auto_canny, avg_len = estimateInstrumentModel(img, info)
    if model[0] == -1:
        return model, start_pt, end_pt, auto_canny, avg_len
    if not info['isEnableRebuild']:
        return model, start_pt, end_pt, avg_len
    shape = img.shape
    t = min(shape[0], shape[1]) * info['rebuildModelDisThreshRatio']
    debug_src = auto_canny.copy()
    debug_src = cv2.cvtColor(debug_src, cv2.COLOR_GRAY2BGR)
    cv2.circle(debug_src, (model[0], model[1]), model[2], color=(255, 0, 0), thickness=1)
    plot.subImage(src=debug_src, index=plot.next_idx(), title="Model")
    saver.saveImg(debug_src, 'approx_model')
    rebuild_lines, start_pt, end_pt = rebuildScaleLines(auto_canny, model, t)
    best_model, _, _ = fitCenter(rebuild_lines, shape, info['rasancDst'])
    # print("Lose :", np.abs(best_model - model))
    return best_model, start_pt, end_pt, avg_len


def analysisConnectedComponentsProps(meter_id, img_dir, config):
    img, info = load(config, img_dir, meter_id)
    roi = meterFinderBySIFT(img, info['template'])
    roi = cv2.GaussianBlur(roi, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    # roi = cv2.fastNlMeansDenoisingColored(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # gray = cv2.fastNlMeansDenoising(gray)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 10)
    # retval, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # thresh = (gray > threshold_otsu(gray)) * 1
    analysis(autoCanny(roi, info), info)


def analysis(src, info):
    debug_src = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    plot.subImage(src=src, index=plot.next_idx(), title='Threshold', cmap='gray')
    auto_canny = LSF.cleanNotInterestedFeatureByProps(src, area_thresh=info['areaThresh'],
                                                      approx_thresh=info['approxThresh'],
                                                      rect_ration_thresh=info['rectRationThresh'])
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Filtered Roughly', cmap='gray')
    auto_canny = cv2.ximgproc.thinning(auto_canny, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Thinning', cmap='gray')
    # auto_canny = cv2.ximgproc.thinning(auto_canny, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # extract scale lines
    # detector = cv2.createLineSegmentDetector()
    # _lines, width, prec, nfa = detector.detect(auto_canny)
    # debug_src = np.zeros([src.shape[0], src.shape[1], 3], dtype=np.uint8)
    # detector.drawSegments(debug_src, _lines)
    # # detector.drawSegments(line_src, _lines)
    # lines, approx_center = LSF.filter(_lines, debug_src.shape)
    # plot.subImage(src=imutils.opencv2matplotlib(debug_src), index=plot.next_idx(), title='Noising Line Scale')
    return


def rebuildScaleLines(auto_canny, model, threshold, start_pt=None, end_pt=None):
    """
    After special image processing, the image may lose some features.We are indeed to protect the scale line feature.
    In this case, scale line reconstruction is needed based on the binary image with the least missing features.
    :param auto_canny:binary image with the least missing features, make hypothesis it contains almost all scale line
           feature
    :param model: model obtained by Ransac fitting algorithm
    :param threshold: distance threshold used to build the descriptors
    :param start_pt: start point coordinate vector of instrument range
    :param end_pt: end point coordinate vector of instrument range
    :return:scale line descriptors
    """
    debug_src = np.zeros([auto_canny.shape[0], auto_canny.shape[1], 3], dtype=np.uint8)
    line_src = debug_src.copy()
    detector = cv2.createLineSegmentDetector()
    lines, width, prec, nfa = detector.detect(auto_canny)
    detector.drawSegments(line_src, lines)
    plot.subImage(src=line_src, index=plot.next_idx(), title="All Lines")
    saver.saveImg(line_src, 'all_lines')
    descriptors, left_lines_set, right_lines_set, line_avg_len = buildLineDescriptors(lines, model,
                                                                                      threshold)
    start_pt, end_pt = calStartEndRange(left_lines_set, right_lines_set, line_avg_len)
    lines = np.array([line[0] for line in descriptors])
    detector.drawSegments(debug_src, lines)
    if start_pt[0] == -1 and end_pt[0] == -1:
        start_pt, end_pt = setDefaultRange(left_lines_set, right_lines_set)
        # raise Exception("Start and end point not found.")
    cv2.circle(debug_src, (start_pt[0], start_pt[1]), 5,
               (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    cv2.circle(debug_src, (end_pt[0], end_pt[1]), 5,
               (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    plot.subImage(src=debug_src, index=plot.next_idx(), title="Rebuild Scale Lines")
    saver.saveImg(debug_src, 'rebuild_scale_lines')
    return lines, start_pt, end_pt


def setDefaultRange(left_lines_set, right_lines_set, start_pt=None, end_pt=None):
    """
    set the default range when cannot find the start and end of the range
    :param left_lines_set:
    :param right_lines_set:
    :param start_pt:
    :param end_pt:
    :return:
    """
    left_lines_set.extend(right_lines_set)
    left_lines_set = sorted(left_lines_set, key=lambda el: el[6])
    right_lines_set = sorted(right_lines_set, key=lambda el: el[6])
    # if len(left_lines_set) > len(right_lines_set):
    start_pt = left_lines_set[0][2]
    end_pt = left_lines_set[len(left_lines_set) - 1][2]
    # else:
    #     start_pt = right_lines_set[0][2]
    #     end_pt = right_lines_set[len(right_lines_set) - 1][2]
    return start_pt, end_pt


def buildLineDescriptors(lines, model, threshold):
    """
    build line descriptors , a descriptor is composed by
    1.line metadata including coords of staring point and end point
    2.line len
    3.line center ,calculated by start pt and end pt
    4.start pt coordinates
    5.end pt coordinates
    6.the line vector angle with reference vector
    :param lines: line metadata returned by LSD'algorithm
    :param model: fitting model by previous Ransac algorithom
    :param threshold: distance thresh to eliminate non-scale-lines
    :return:
    """
    descriptors = []
    line_avg_len = 0
    model_center = [model[0], model[1]]
    radius = model[2]
    vertical_vector = np.array([0, 1])
    left_lines_set = []
    right_lines_set = []
    for index, line in enumerate(lines):
        l = line[0]
        line_center = (np.array([l[0], l[1]]) + np.array([l[2], l[3]])) / 2
        # scale line intersect with circle
        start = np.array([l[0], l[1]])
        end = np.array([l[2], l[3]])
        # distance between scale line intersection pt and model center.
        insec_with_model = LU.getDistPtToLine(model_center, start, end) < threshold
        # distance between scale line center and circle model margin.
        center_in_model = np.abs(EuclideanDistance(model_center, line_center) - model[2]) < threshold
        line_len = EuclideanDistance(start, end)
        # line thresh for filtering len line ,which don't belong to scale line obviously
        is_len_proper = line_len < radius * 0.25
        if insec_with_model and center_in_model and is_len_proper:
            line_avg_len += line_len
            angle, is_left, line_vector = buildLineVector(start, end, model_center, vertical_vector)
            descriptor = [line, line_len, line_center, start, end, line_vector, angle]
            if is_left:
                left_lines_set.append(descriptor)
            else:
                right_lines_set.append(descriptor)
            descriptors.append(descriptor)
    len_des = len(descriptors)
    if len_des == 0:
        print("Not found line descriptors.")
    line_avg_len /= len(descriptors)
    return descriptors, left_lines_set, right_lines_set, line_avg_len


def buildLineVector(start, end, model_center, reference):
    """
    calculate angle between line vector and reference vector
    :param start: line starting point
    :param end:  line end point
    :param model_center: model center for calculating distance
    :param reference: vector own reference property
    :return:angle ,a boolean to judge line whether is in the left of reference or not,line vector
    """
    line_vector = getLineVector(start, end, model_center)
    angle, is_left = calClockwiseAngleWithReferenceVector(line_vector, reference)
    return angle, is_left, line_vector


def calClockwiseAngleWithReferenceVector(line_vector, reference_vector):
    """
    calculate clockwise vector with reference vector,
    if angle is greater than np.pi ,substract np.pi for
    obtaining symmetrical scale lines at the forward algorithm procedure
    :param line_vector:
    :param reference_vector:
    :return: angle
    """
    angle = AngleFactory.calAngleClockwiseByVector(reference_vector, line_vector)
    if angle <= np.pi:
        return angle, True
    else:
        angle = np.pi * 2 - angle
        return angle, False


def calStartEndRange(left_lines_set, right_lines_set, len_thresh, start_pt=None, end_pt=None):
    left_lines_set = sorted(left_lines_set, key=lambda el: el[6])
    right_lines_set = sorted(right_lines_set, key=lambda el: el[6])
    start_pt = np.array([-1, -1])
    end_pt = np.array([-1, -1])
    for el_left in left_lines_set:
        is_found = False
        for el_right in right_lines_set:
            angle_in_range = np.rad2deg(np.abs(el_left[6] - el_right[6])) < 2
            len_in_range = el_right[1] > len_thresh * 0.9 and el_left[1] > len_thresh * 0.9
            if angle_in_range and len_in_range:
                start_pt = el_left[2]
                end_pt = el_right[2]
                is_found = True
                break
        if is_found:
            break
    return start_pt, end_pt


def getLineVector(start, end, model_center, line_vector=None):
    """
    :param start: line staring point
    :param end: line end point
    :param model_center: model center
    :param line_vector: vector ,whose orientation is from center point to outline.
    :return:
    """
    pt1_ds = EuclideanDistance(start, model_center)
    pt2_ds = EuclideanDistance(end, model_center)
    # find farthest point
    if pt1_ds > pt2_ds:
        line_vector = start - end
    else:
        line_vector = end - start
    return np.array(line_vector)


def list_bisection_search(list, e, lo, hi=None):
    if hi is None:
        hi = len(list)
    if lo < 0:
        raise ValueError("Low boundary cannot least 0.")
    while lo < hi:
        mi = (lo + hi) // 2
        if e < list[mi]:
            hi = mi
        elif list[mi] < e:
            lo = mi + 1
        else:
            return mi
    return -1


def estimateInstrumentModel(src, info, rough_lines=None):
    """
    estimate a instrument model
    :param src:
    :param info:
    :param rough_lines:
    :return:
    """
    # src = meterFinderByTemplate(image, info["template"])
    auto_canny = autoCanny(src, info)
    rough_lines = extractRoughScaleLines(auto_canny, info)
    model, line_centers, inliers_idx = fitCenter(rough_lines, info['template'].shape, info['rasancDst'])
    if model[0] == -1:
        return model, [-1, -1], [-1, -1], None, -1
    center = [model[0], model[1]]
    vertical_vector = np.array([0, 1])
    descriptors = []
    avg_len = 0
    for idx, line in enumerate(rough_lines):
        # determine if the idx is in inliers or not
        if list_bisection_search(inliers_idx, idx, 0) == -1:
            continue
        # is inliers
        line = line[0]
        start = np.array([line[0], line[1]])
        end = np.array([line[2], line[3]])
        avg_len += EuclideanDistance(start, end)
        angle, _, _, = buildLineVector(start, end, center, vertical_vector)
        descriptors.append([line_centers[idx], angle])
    descriptors = sorted(descriptors, key=lambda el: el[1])
    # start point and end point's coordinates are estimated,could be selectively use.
    if len(descriptors) == 0:
        raise ValueError('Fit failed.')
    avg_len /= len(inliers_idx)
    start_pt = np.int32(descriptors[0][0])
    end_pt = np.int32(descriptors[len(descriptors) - 1][0])
    return model, start_pt, end_pt, auto_canny, avg_len


# src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0)
# src = cv2.fastNlMeansDenoisingColored(src, h=7, templateWindowSize=7, searchWindowSize=21)

# Auto Canny + Mask


def extractRoughScaleLines(src, info):
    saver.saveImg(src, 'before_cleaning_noise')
    auto_canny = LSF.cleanNotInterestedFeatureByProps(src, area_thresh=info['areaThresh'],
                                                      approx_thresh=info['approxThresh'],
                                                      rect_ration_thresh=info['rectRationThresh'])
    # auto_canny = LSF.cleanNotInterestedFeature(src)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Binary Src After cleaning', cmap='gray')
    saver.saveImg(auto_canny, 'after_cleaning_noise')
    if info['enableLineThinning']:
        auto_canny = cv2.ximgproc.thinning(auto_canny, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        plot.subImage(src=auto_canny, index=plot.next_idx(), title='Thinning', cmap='gray')
        saver.saveImg(auto_canny, 'thinning_auto_canny')
    detector = cv2.createLineSegmentDetector()
    # extract all lines contours using LSD algorithm
    _lines, width, prec, nfa = detector.detect(auto_canny)
    # only consider the points in image range
    # _lines = np.array(
    #    [_line for _line in _lines if _line[0][0] > 0 and _line[0][1] and _line[0][2] > 0 and _line[0][3] > 0])
    debug_src = np.zeros([src.shape[0], src.shape[1], 3], dtype=np.uint8)
    detector.drawSegments(debug_src, _lines)
    plot.subImage(src=imutils.opencv2matplotlib(debug_src), index=plot.next_idx(),
                  title='Line Segments Detected by LSD algorithm')
    saver.saveImg(debug_src, 'line_segments_detected_by_LSD')
    debug_src = np.zeros([src.shape[0], src.shape[1], 3], dtype=np.uint8)
    # detector.drawSegments(line_src, _lines)
    # double match lines in accordance with the LSD functionality
    # which splits a linear contour to two thin line
    lines, approx_center = LSF.matchScaleLines(_lines, debug_src.shape)
    detector.drawSegments(debug_src, lines)
    plot.subImage(src=imutils.opencv2matplotlib(debug_src), index=plot.next_idx(), title='Noising Line Scale')
    saver.saveImg(debug_src, 'matched_scale_lines')
    return lines


def autoCanny(src, info):
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray)
    gray_test = gray.copy()
    kernel_size = info['kernelSize']
    # plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=plot.next_idx(), title='Fast denosing')
    # retval, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # gray_test, covex_mask = LSF.filterContex(gray_test)
    # Tresh + Gray Convex Masx
    # thresh_pre = cv2.bitwise_and(thresh, covex_mask)
    # plot.subImage(src=thresh_pre, index=plot.next_idx(), title='lhsh conver', cmap='gray')
    # thresh_convex, tm = LSF.filterContex(thresh)
    # plot.subImage(src=thresh_convex, index=plot.next_idx(), title='Thresh conver', cmap='gray')
    # plot.subImage(src=tm, index=plot.next_idx(), title='Thresh tm', cmap='gray')
    # plot.subImage(src=thresh, index=plot.next_idx(), title='Thresh_OTSU', cmap='gray')
    auto_canny = imutils.auto_canny(gray)
    dilate_kernel = cv2.getStructuringElement(ksize=(kernel_size, kernel_size), shape=cv2.MORPH_ELLIPSE)
    erode_kernel = cv2.getStructuringElement(ksize=(kernel_size, kernel_size), shape=cv2.MORPH_ELLIPSE)
    # fill scale line with white pixels
    auto_canny = cv2.dilate(auto_canny, dilate_kernel)
    auto_canny = cv2.erode(auto_canny, erode_kernel)
    plot.subImage(src=auto_canny, index=plot.next_idx(), title='Auto Canny', cmap='gray')
    saver.saveImg(auto_canny, 'auto_canny')
    return auto_canny


def fitCenter(lines, shape, dst_thresh):
    detector = cv2.createLineSegmentDetector()
    # debug image
    debug_src = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
    detector.drawSegments(debug_src, lines)
    dst_thresh = min(shape[0], shape[1]) * 0.01
    # print("Ransac Thresh", dst_thresh)
    # compose a proper format for RASANC algorithm
    line_centers = [np.array([(l[0][0] + l[0][2]) / 2, (l[0][1] + l[0][3]) / 2]) for l in lines]
    optimal = np.round(len(line_centers) * 0.9)  # expected optimal result
    # use random sample consensus to fit a best circle model
    best_circle, max_fit_num, inliers_idx = rasan.randomSampleConsensus(data=line_centers,
                                                                        max_iterations=200,
                                                                        optimal_consensus_num=optimal,
                                                                        dst_threshold=dst_thresh)
    best_circle = np.int32(best_circle)
    if best_circle[0] == -1:
        return best_circle, [], []
    for idx, c in enumerate(line_centers):
        c = np.int32(c)
        # display inliers points and lines with center
        if list_bisection_search(inliers_idx, idx, 0) != -1:
            cv2.circle(debug_src, (np.int32(c[0]), np.int32(c[1])), 3, (0, 255, 0), cv2.FILLED)
            cv2.line(debug_src, (best_circle[0], best_circle[1]), (c[0], c[1]), color=(255, 0, 0), thickness=1)
        # display outlier points
        else:
            cv2.circle(debug_src, (np.int32(c[0]), np.int32(c[1])), 3, (255, 255, 255), cv2.FILLED)
    # display model center
    cv2.circle(debug_src, (best_circle[0], best_circle[1]), 10, color=(0, 0, 255), thickness=cv2.FILLED)
    plot.subImage(src=imutils.opencv2matplotlib(debug_src), index=plot.next_idx(), title='Fitting Circle Model')
    saver.saveImg(debug_src, 'fitting_circle')
    return best_circle, line_centers, inliers_idx


def enhance(src):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    normalize = np.zeros(gray.shape)
    cv2.normalize(gray, normalize, 0, 1, cv2.NORM_MINMAX)
    # plot.subImage(src=normalize, index=plot.next_idx(), title='Gray', cmap='gray')
    # plot.subImage(src=gray, index=plot.next_idx(), title='Gray', cmap='gray')
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    laplace = cv2.filter2D(gray, cv2.CV_8UC1, laplace_kernel)
    # laplace = cv2.Laplacian(gray, cv2.CV_8UC1, ksize=3)
    saver.saveImg(laplace, 'enhance_laplace')
    # abs_labplace = cv2.convertScaleAbs(laplace)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Laplace', cmap='gray')
    # cv2.normalize(abs_labplace, abs_labplace)
    overlay = cv2.add(gray, laplace)
    saver.saveImg(overlay, 'enhance_gray_sub_laplace')
    # plot.subImage(src=overlay, index=plot.next_idx(), title='Overlay by Gray + Laplace', cmap='gray')
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    grad = cv2.add(sobel_x, sobel_y)
    # cv2.normalize(grad, grad, 1.0, 0.0, cv2.NORM_MINMAX)
    # plot.subImage(src=grad, index=plot.next_idx(), title='Grad', cmap='gray')
    media_blur = cv2.medianBlur(grad, 5)
    saver.saveImg(media_blur, 'enhance_grad_media_blur')
    # media_blur = np.float32(media_blur) * (1 / 255)
    # overlay = np.float32(overlay) * (1 / 255)
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Media Blurred', cmap='gray')
    mask = cv2.bitwise_and(media_blur, overlay)
    # mask = (mask > 1) * 255
    # np.where(mask >= 1, 255, 0)
    #  print(mask)
    saver.saveImg(mask, 'enhance_mask')
    # plot.subImage(src=abs_labplace, index=plot.next_idx(), title='Mask', cmap='gray')
    enhance = cv2.add(gray, mask)
    saver.saveImg(enhance, 'enhanced')
    # plot.subImage(src=enhance, index=plot.next_idx(), title='Enhance', cmap='gray')
    return cv2.cvtColor(enhance, cv2.COLOR_GRAY2BGR)


def test_enhancement():
    #
    img = cv2.imread('image/1-1.jpg')
    # img = cv2.imread('image/WechatIMG40.png')
    template = cv2.imread('template/1-1_1.jpg')
    src = meterFinderBySIFT(img, template)
    enhance(src)


if __name__ == '__main__':
    #     # res1 = readPressureValueFromDir('lxd1_4', 'image/lxd1.jpg', 'config/lxd1_4.json')
    #     # res2 = readPressureValueFromDir('szk2_1', 'image/szk2.jpg', 'config/szk2_1.json')
    #     # res3 = readPressureValueFromDir('szk1_5', 'image/szk1.jpg', 'config/szk1_5.json')
    #     # res4 = readPressureValueFromDir('wn1_5', 'image/wn1.jpg', 'config/wn1_5.json')
    #     # res5 = readPressureValueFromDir('xyy3_1', 'image/xyy3.jpg', 'config/xyy3_1.json')
    #     # res6 = readPressureValueFromDir('pressure2_1', 'image/pressure2.jpg', 'config/pressure2_1.json')
    #     # init('pressure2_1', 'image/pressure2.jpg', 'config/pressure2_1.json')
    #     # analysisConnectedComponentsProps('pressure2_1', 'image/pressure2.jpg', 'config/pressure2_1.json')
    img_main_dir = 'image/pointer'
    images = os.listdir(img_main_dir)
    config = os.listdir("config")
    for im in images:
        img_dir = img_main_dir + os.path.sep + im
        for i in range(1, 6):
            meter_id = im.split(".jpg")[0] + "_" + str(i)
            cfg_dir = meter_id + '.json'
            if cfg_dir in config:
                start = time.time()
                load(meter_id, img_dir, 'config/' + cfg_dir)
                # print("Time consumption: ", end - start)
#     # try:
#     #     # analysisConnectedComponentsProps('lxd1_2', 'image/lxd1.jpg', 'config/lxd1_2.json')
#     #     # initExtractScaleLine('lxd1_2', 'image/lxd1.jpg', 'config/lxd1_2.json')
#     #     start = cv2.getTickCount()
#     res = load('pressure2_1', 'image/pressure2.jpg', 'config/pressure2_1.json')
#     res2 = load('lxd1_2', 'image/lxd1.jpg', 'config/lxd1_2.json')
#     res3 = load('lxd2_1', 'image/lxd2.jpg', 'config/lxd2_1.json')
#     res4 = load('lxd3_1', 'image/lxd3.jpg', 'config/lxd3_1.json')
#     res5 = load('1-1_1', 'image/1-1.jpg', 'config/1-1_1.json')
#     res6 = load('1-1_2', 'image/1-1.jpg', 'config/1-1_2.json')
#     res7 = load('1-2_1', 'image/1-2.jpg', 'config/1-2_1.json')
#     res8 = load('1-2_2', 'image/1-2.jpg', 'config/1-2_2.json')
# #     # test_enhancement()
#     t = (cv2.getTickCount() - start) / cv2.getTickFrequency()
#     print("Time consumption: ", t)
# finally:
#     print(res)
#     # print(res2)
#     # print(res3)
#     # print(res5)
#     #  print(res6)
#     # print(res8)
#     plot.show(save=True)
