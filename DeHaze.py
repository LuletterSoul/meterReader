import optparse
import cv2
import numpy as np
import sys


def create_dehaze_options():
    usage = "usage: %prog <options>"
    parser = optparse.OptionParser(prog='dehaze', usage=usage)
    parser.add_option('-i', '--image', type='string', dest='image', help='Image to dehaze')
    parser.add_option('-o', '--output', type='string', dest='output', help='Path to save the output image')
    return parser


def zmMinFilterGray(src, r=7):
    '''minimum filter with radius'''
    if r <= 0:
        return src
    h, w = src.shape[: 2]
    I = src
    res = np.minimum(I, I[[0] + list(range(h - 1)), :])
    res = np.minimum(res, I[list(range(1, h)) + [h - 1], :])
    I = res
    res = np.minimum(I, I[:, [0] + list(range(w - 1))])
    res = np.minimum(res, I[:, list(range(1, w)) + [w - 1]])
    return zmMinFilterGray(res, r - 1)


def guidedfilter(I, p, r, eps):
    '''Guided Filter'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p
    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I
    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I
    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):
    '''
    m is a RGB image with color value ranged in [0, 1]
    Computing transmittance mask image V1
    and Airlight A
    '''
    # Dark Channel Image
    V1 = np.min(m, 2)

    # Apply Guided Filter
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)

    # Calculating Airlight
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    # Limit value range
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    # Calculating transmittance mask and airlight
    V1, A = getV1(m, r, eps, w, maxV1)
    for k in range(3):
        # Color correction
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)
        Y = np.clip(Y, 0, 1)

        # Gamma correction if required
        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y


if __name__ == '__main__':
    parser = create_dehaze_options()
    (option, args) = parser.parse_args()

    if option.image == None or option.output == None:
        print(parser.format_help())
        sys.exit(1)

    # 读取需要去雾的图片
    # 并将所有的值映射到[0, 1]内
    haze = cv2.imread(option.image) / 255.0

    # 应用暗通道先验去雾算法
    # 重新将色彩的值映射回[0, 255]
    dehazed = deHaze(haze) * 255

    # 输出图像
    cv2.imwrite(option.output, dehazed)
