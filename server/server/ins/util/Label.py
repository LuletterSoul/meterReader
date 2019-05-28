from .ROIUtil import *
import cv2
import os
import json


def setCoordinates():
    config_dir = os.path.join('labels', 'an')
    dirs = os.listdir(config_dir)
    startPoint = [237, 772]
    centerPoint = [503, 521]
    endPoint = [772, 776]
    for d in dirs:
        dir = config_dir + os.path.sep + d
        print(dir)
        config = open(dir)
        josnc = json.load(config)
        josnc["startPoint"]["x"] = startPoint[0]
        josnc["startPoint"]["y"] = startPoint[1]
        josnc["centerPoint"]["x"] = centerPoint[0]
        josnc["centerPoint"]["y"] = centerPoint[1]
        josnc["endPoint"]["x"] = endPoint[0]
        josnc["endPoint"]["y"] = endPoint[1]
        config.write(json.dumps(josnc, indent=4))


def formatConfig(config_dir, template_dir, template_output_dir, src_output_dir):
    dirs = os.listdir(config_dir)
    for d in dirs:
        template_cfg = open(template_dir)
        # 按模板配置新建一个json对象
        template_json = json.load(template_cfg)
        filename, extention = os.path.splitext(d)
        if extention != '.json':
            continue
        src_dir = config_dir + os.path.sep + d
        print('Processing: ', src_dir)
        # 源配置
        sir_cfg = open(src_dir)
        src_json = json.load(sir_cfg)
        # 先读模板，按照模板键值出现的顺序进行格式化
        for key in template_json:
            # 如果源配置的键值在模板中，将对应的值保存
            if key in src_json:
                template_cfg.key = src_json.key
            # 如果不在，将不存在的键值保存在源配置中
            else:
                src_json.key = template_json.key
        src_output = open(src_output_dir, 'w')
        template_output = open(template_output_dir, 'w')
        # 保存到指定目录
        src_output.write(json.dumps(src_json, indent=4))  # indent =4  保留json格式
        template_output.write(json.dumps(template_json, indent=4))


def modifyConfig():
    #  config_dir = os.path.join('labels', 'an')
    config_dir = 'config'
    dirs = os.listdir(config_dir)
    for d in dirs:
        filename, extention = os.path.splitext(d)
        if extention != '.json':
            continue
        dir = config_dir + os.path.sep + d
        print(dir)
        config = open(dir)
        josnc = json.load(config)
        if josnc['type'] == 'normalPressure':
            config.close()
            config = open(dir, 'w')
            josnc['ptRegAlgType'] = 0
            josnc['matchTemplateType'] = 2
            # josnc['enableLineThinning'] = False
            # josnc['ptRegAlgType'] = 1
            josnc['enableFitting'] = False
            # josnc['enableFitting'] = True
            config.write(json.dumps(josnc, indent=4))


def formatConfig(config_dir, ptRegAlgType=0, enableFitting=False):
    #  config_dir = os.path.join('labels', 'an')
    filename, extention = os.path.splitext(os.path.basename(config_dir))
    if extention == '.json':
        config = open(config_dir)
        josnc = json.load(config)
        if josnc['type'] == 'normalPressure':
            config.close()
            config = open(config_dir, 'w')
            josnc['ptRegAlgType'] = ptRegAlgType
            josnc['matchTemplateType'] = 2
            # josnc['enableLineThinning'] = False
            # josnc['ptRegAlgType'] = 1
            josnc['enableFitting'] = enableFitting
            # josnc['enableFitting'] = True
            config.write(json.dumps(josnc, indent=4))


if __name__ == '__main__':
    # setCoordinates()
    modifyConfig()

# if __name__ == '__main__':
#     img_dir = os.path.join('image', 'anpressure')
#     dirs = os.listdir(img_dir)
#     info = json.load(open('config/template.json'))
#     for d in dirs:
#         name, extention = os.path.splitext(d)
#         img = img_dir + os.path.sep + d
#         print("Label image:", img)
#         img = cv2.imread(img)
#         if img is None:
#             continue
#         r = selectROI(img)
#         if 0 == len(r):
#             continue
#         info['name'] = name
#         info['type'] = 'normalPressure'
#         info['ROI'] = {
#             'x': int(r[0][0]),
#             'y': int(r[0][1]),
#             'w': int(r[0][2]),
#             'h': int(r[0][3])
#         }
#         print(info)
#         label_dir = 'labels' + os.path.sep + name + '.json'
#         file = open(label_dir, "w")
#         file.write(json.dumps(info, indent=4))
#         print("Label success")
