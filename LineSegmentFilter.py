from Common import *
import imutils

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


def clean(src, new_src=None):
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
        if len(approx) <= 8 and cv2.contourArea(contour) < 500:
            cv2.drawContours(new_src, [contour], -1, 255, cv2.FILLED)
    return new_src


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

    def filter(binary, draw_src):
        line_detector = cv2.createLineSegmentDetector()
        # Returned Vec4i [x1,y1,x2,y2],(x1,y1) is the start,(x2,y2) is the end.
        _lines, width, prec, nfa = line_detector.detect(binary)
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

        match_pair = []
        line_set_len = len(line_descriptors)
        vis = np.zeros(shape=(line_set_len, 1), dtype=np.int32)
        avg_center_dis = 0
        avg_diff_angle = 0
        avg_diff_len = 0
        index_record = []
        for i in range(0, line_set_len):
            if vis[i] == 1:
                continue
            for j in range(0, line_set_len):
                if i == j or vis[j] == 1:
                    continue
                t = cv2.getTickCount()
                dis_center = EuclideanDistance(line_descriptors[i][3], line_descriptors[j][3])
                hor_angle1 = AngleFactory.calAngleClockwiseByVector(level_line_vector, line_descriptors[i][2])
                hor_angle2 = AngleFactory.calAngleClockwiseByVector(level_line_vector, line_descriptors[j][2])
                # fix vector orientation in order to locate angle range in [0,pi]
                if hor_angle1 >= 1.5 * np.pi:
                    hor_angle1 = hor_angle1 - np.pi
                if hor_angle2 >= 1.5 * np.pi:
                    hor_angle2 = hor_angle2 - np.pi
                diff_angle = np.rad2deg(np.abs(hor_angle2 - hor_angle1))
                # print("Count: " + str(i * j) + " : " + str(diff_angle))
                diff_len = np.abs(line_descriptors[i][4] - line_descriptors[j][4])
                # avg_center_dis += dis_center
                # avg_diff_angle += diff_angle
                # avg_diff_len += diff_len
                n = (cv2.getTickCount() - t) / cv2.getTickFrequency()
                # print("Execution Time of Per iteration: ", n)
                if dis_center < thresh1 and diff_len < thresh2 and diff_angle < thresh3:
                    match_pair.append(_lines[i])
                    match_pair.append(_lines[j])
                    vis[i] = 1
                    vis[j] = 1
                    end = (cv2.getTickCount() - t) / cv2.getTickFrequency()
                    print("Execution Time of Per match iteration:", end)
                    break

        avg_div = line_set_len * (line_set_len - 1)
        print("Avg center distance:", avg_center_dis / avg_div)
        print("Avg ang:", avg_diff_angle / avg_div)
        print("Avg len:", avg_diff_len / avg_div)
        #  print([np.array([l[0][0], l[1][1], l[1][0], l[1][1]]) for l in match_pair if len(l) > 0])
        line_detector.drawSegments(draw_src, np.array(match_pair))
        # line_detector.drawSegments(draw_src, _lines)


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
