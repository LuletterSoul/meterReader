from util.ROIUtil import *
import cv2
import os
import json

if __name__ == '__main__':
    img_dir = os.path.join('image', 'anpressure')
    dirs = os.listdir(img_dir)
    info = json.load(open('config/template.json'))
    for d in dirs:
        name, extention = os.path.splitext(d)
        img = img_dir + os.path.sep + d
        print("Label image:", img)
        img = cv2.imread(img)
        if img is None:
            continue
        r = selectROI(img)
        if 0 == len(r):
            continue
        info['name'] = name
        info['type'] = 'normalPressure'
        info['ROI'] = {
            'x': int(r[0][0]),
            'y': int(r[0][1]),
            'w': int(r[0][2]),
            'h': int(r[0][3])
        }
        print(info)
        label_dir = 'labels' + os.path.sep + name + '.json'
        file = open(label_dir, "w")
        file.write(json.dumps(info, indent=4))
        print("Label success")
