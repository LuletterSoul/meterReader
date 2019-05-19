from server.server.ins.Common import *
import os
import time
import imutils
from server.server.ins.util.StoreUtil import DataSaver


def enhancing(meter_id, img_main_dir, img_dir, template_dir, output_dir, config):
    img = cv2.imread(img_dir)
    if img is None:
        print("Image cannot be none.")
        return -1
    # roi = meterFinderBySIFT2(r, info['template'], info)
    fast = cv2.fastNlMeansDenoisingColored(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    save_dir = output_dir + os.path.sep + meter_id + '_6_binary' + '.png'
    binary = imutils.auto_canny(gray)
    cv2.imwrite(save_dir, binary)
    save_dir = output_dir + os.path.sep + meter_id + '_7_enhance_binary' + '.png'
    en_img = enhance(fast, meter_id, output_dir)
    en_binary = imutils.auto_canny(en_img)
    dilate = cv2.dilate(en_binary, (5, 5))
    en_binary = cv2.erode(dilate, (5, 5))
    cv2.imwrite(save_dir, dilate)
    cv2.imwrite(meter_id + '.png', img)


def denoising(meter_id, img_main_dir, img_dir, template_dir, output_dir, config):
    img = cv2.imread(img_dir)
    saver = DataSaver(output_dir)
    if img is None:
        print("Image cannot be none.")
    # roi = meterFinderBySIFT2(r, info['template'], info)
    kernel_size = 5
    kernel = (kernel_size, kernel_size)
    mean = cv2.blur(img, kernel)
    gassian = cv2.GaussianBlur(img, kernel, 0)
    fast = cv2.fastNlMeansDenoisingColored(img)
    str_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
    # original = output_dir + os.path.sep + meter_id + '_origin_' + str_time + '.png'
    # mean_out = output_dir + os.path.sep + meter_id + '_mean_' + str_time + '.png'
    # gassian_out = output_dir + os.path.sep + meter_id + '_gassian_' + str_time + '.png'
    # fast_out = output_dir + os.path.sep + meter_id + '_fast_' + str_time + '.png'
    original = output_dir + os.path.sep + meter_id + '_origin' + '.png'
    mean_out = output_dir + os.path.sep + meter_id + '_mean' + '.png'
    gassian_out = output_dir + os.path.sep + meter_id + '_gassian' + '.png'
    fast_out = output_dir + os.path.sep + meter_id + '_fast' + '.png'
    cv2.imwrite(original, img)
    cv2.imwrite(mean_out, mean)
    cv2.imwrite(gassian_out, gassian)
    cv2.imwrite(fast_out, fast)


def initDenoisingTest():
    global i
    current_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    img_main_dir = os.path.abspath(current_path + "../image/denoising")
    output_main_dir = os.path.abspath(current_path + "../data/denoising")
    images = os.listdir(img_main_dir)
    for im in images:
        img_dir = img_main_dir + os.path.sep + im
        im = im.lower()
        meter_id, extention = os.path.splitext(im.lower())
        if extention == '.jpg' or extention == '.png':
            # denoising(meter_id, img_main_dir, img_dir, None, output_main_dir, None)
            enhancing(meter_id, img_main_dir, img_dir, None, output_main_dir, None)


if __name__ == '__main__':
    initDenoisingTest()
