from Common import *

thresh1 = 20
thresh2 = 25
thresh3 = 10


def calLineVector(line_a):
    return np.array([line_a[2] - line_a[0], line_a[3] - line_a[1]])


def calCenter(line_a):
    return np.array([(line_a[0] + line_a[2]) / 2, (line_a[1] + line_a[3]) / 2])


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
            if dis_center < thresh1 and diff_len < thresh2 and diff_angle < thresh3:
                match_pair.append(_lines[i])
                match_pair.append(_lines[j])
                vis[i] = 1
                vis[j] = 1
                break
    avg_div = line_set_len * (line_set_len - 1)
    print("Avg center distance:", avg_center_dis / avg_div)
    print("Avg ang:", avg_diff_angle / avg_div)
    print("Avg len:", avg_diff_len / avg_div)
    #  print([np.array([l[0][0], l[1][1], l[1][0], l[1][1]]) for l in match_pair if len(l) > 0])
    line_detector.drawSegments(draw_src, np.array(match_pair))
    # line_detector.drawSegments(draw_src, _lines)
