from Common import *
import json
import os


def shift(meter_id, img_dir, template_dir, output_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    info["template"] = cv2.imread(template_dir)
    print("Img: ", meter_id)
    # ROI extract
    x = info["ROI"]["x"]
    y = info["ROI"]["y"]
    w = info["ROI"]["w"]
    h = info["ROI"]["h"]
    r = img[y:y + h, x:x + w]
    roi_dir = 'data/shift/' + meter_id + '_roi' + '.png'
    cv2.imwrite(roi_dir, r)
    roi = meterFinderBySIFT(r, info['template'], info)
    shift_dir = output_dir + os.path.sep + meter_id + '.png'
    cv2.imwrite(shift_dir, roi)


if __name__ == '__main__':
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
