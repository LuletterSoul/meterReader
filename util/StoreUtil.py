import time
import cv2
import os
import json
from openpyxl import *
import csv
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

    def saveConfig(self, info):
        """
        备份指定meterId的配置文件
        :param meter_id:
        :param config_base_dir:
        :return:
        """
        save_config_path = os.path.join(self.save_path, self.meter_id + ".json")
        info['template'] = None
        if 'saver' in info and info['saver'] is not None:
            info['saver'] = None
        if not os.path.exists(save_config_path):
            config = open(save_config_path, "w")
            config.write(json.dumps(info, indent=4))


def saveToExcelFromDic(excel_dir, dic_content):
    wb = load_workbook(excel_dir)
    str_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    ws = wb.create_sheet(title=str_time)
    for content in dic_content:
        ws.append(list(content.values()))
    wb.save(excel_dir)


# if __name__ == '__main__':
#     data = [{'meterId': '3-14_1', 'imageKeyPointNum': 3063, 'templateKeyPointNum': 1669, 'realValue': 0.45,
#              'readingValue': 0.445, 'absError': '0.005 %'},
#             {'meterId': '3-14_1', 'imageKeyPointNum': 3063, 'templateKeyPointNum': 1669, 'realValue': 0.45,
#              'readingValue': 0.445, 'absError': '0.005 %'}]
#     current_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
#     main_dir = os.path.abspath(current_path + "../data")
#     saveToExcelFromDic(main_dir + '/output.xlsx', data)
