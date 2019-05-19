from server.server.ins.Common import *
import json
import os
import time
from server.server.ins.util.StoreUtil import DataSaver


def shift(meter_id, img_dir, template_dir, output_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    str_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
    saver = DataSaver(output_dir + os.path.sep, meter_id)
    info['saver'] = saver
    info["template"] = cv2.imread(template_dir)
    if img is None or info['template'] is None:
        print("Image cannot be none.")
    print("Img: ", meter_id)
    # ROI extract
    x = info["ROI"]["x"]
    y = info["ROI"]["y"]
    w = info["ROI"]["w"]
    h = info["ROI"]["h"]
    r = img[y:y + h, x:x + w]
    roi_dir = 'data/shift/' + meter_id + '_roi' + '.png'
    cv2.imwrite(roi_dir, r)
    # roi = meterFinderBySIFT2(r, info['template'], info)
    roi = meterFinderBySIFT(r, info['template'], info)
    saver.saveImg(roi, 'roi')
    # roi = meterFinderBySIFT(r, info['template'], info)


def noisyTest():
    current_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    img_main_dir = os.path.abspath(current_path + "../image/all")
    output_main_dir = os.path.abspath(current_path + "../data/")
    images_dir = os.listdir(img_main_dir)
    saver = DataSaver(output_main_dir + os.path.sep, 'noisy')
    for im in images_dir:
        im = im.lower()
        prefix = im.split(".jpg")[0]
        img_dir = img_main_dir + os.path.sep + im
        print(img_dir)
        src = cv2.imread(img_dir)
        if src is None:
            continue
        # noised = noisy('gauss', src)
        saver.saveImg(src, prefix + '_src')
        # saver.saveImg(noised, prefix + '_noised')
        # denoised = cv2.fastNlMeansDenoisingColored(noised)
        denoised = cv2.fastNlMeansDenoisingColored(src)
        saver.saveImg(denoised, prefix + '_denoised')


def initShift():
    global i
    current_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    img_main_dir = os.path.abspath(current_path + "../image/anpressure")
    config_main_dir = os.path.abspath(current_path + "../labels/an")
    output_main_dir = os.path.abspath(current_path + "../data/shift")
    template_main_dir = os.path.abspath(current_path + "../template/an")
    images = os.listdir(img_main_dir)
    config = os.listdir(config_main_dir)
    for im in images:
        img_dir = img_main_dir + os.path.sep + im
        for i in range(1, 6):
            meter_id = im.split(".JPG")[0] + "_" + str(i)
            template_dir = template_main_dir + os.path.sep + meter_id + ".jpg"
            cfg_dir = meter_id + '.json'
            if cfg_dir in config:
                shift(meter_id, img_dir, template_dir, output_main_dir, config_main_dir + os.path.sep + cfg_dir)


if __name__ == '__main__':
    # initShift()
    noisyTest()
