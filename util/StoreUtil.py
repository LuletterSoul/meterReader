import time
import cv2
import os
from DebugSwitcher import is_save


class DataSaver:
    def __init__(self, data_dir=None, meter_id=None):
        if data_dir is None or meter_id is None:
            return
        self.data_dir = data_dir
        self.meter_id = meter_id
        meter_dir = data_dir + meter_id
        if not os.path.exists(meter_dir):
            os.mkdir(meter_dir)
        str_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
        current_data_dir = meter_dir + os.path.sep + str(len(os.listdir(meter_dir)) + 1) + '_' + str_time
        os.mkdir(current_data_dir)
        self.save_path = current_data_dir

    def saveImg(self, img, name):
        if not is_save:
            return False
        img_dir = self.save_path + os.path.sep + str(len(os.listdir(self.save_path))) + '_' + name + '.png'
        cv2.imwrite(img_dir, img)
        return True
