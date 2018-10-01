# This script should be placed in the parent folders of all dataset folders
import os
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt

def is_intersect(rec1, rec2):
    # 0 - left/xmin, 1 - right/xmax, 2 - top/ymin, 3 - bot/ymax
    if (rec1[0] > rec2[1]) or (rec1[1] < rec2[0]):
        return False
    if (rec1[2] > rec2[3]) or (rec1[3] < rec2[2]):
        return False
    return True

def overlap(rec1, rec2):
    dx = min(rec1[1], rec2[1]) - max(rec1[0], rec2[0])
    dy = min(rec1[3], rec2[3]) - max(rec1[2], rec2[2])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def total(rec1, rec2):
    area1 = (rec1[1]-rec1[0])*(rec1[3]-rec1[2])
    area2 = (rec2[1]-rec2[0])*(rec2[3]-rec2[2])
    return area1 + area2 - overlap(rec1, rec2)

path = os.getcwd()+"/"
# key is dataset name, value is a list 
# with first element being the image folder, i.e. where the original images are
# and second being the label folder, i.e. where the segmented images are
#datasets = {"etrims-db_v1/": ["images/08_etrims-ds/", "annotations/08_etrims-ds/"]}
models = ["normal_yolo/", "tiny_yolo/"]
datasets = ["ruemonge428/", "etrims-db_v1/", "cvpr2010/"]
fig = plt.figure()
pos = 0
for model in models:
    for dataset in datasets:
        print(model + " && " + dataset)
        # read bounding box info from .txt files into list first
        # then store them in dictionary
        # format - key: name of a image; value: [[left right top bot x y prob], [box2], ...]
        with open(path+dataset+model+"predictions.txt", 'r') as f:
            all_pred = f.read().splitlines()
        with open(path+dataset+model+"truth.txt", 'r') as f:
            all_truth = f.read().splitlines()
        pred_boxes = {}
        truth_boxes = {}
        for i in all_pred:
            tmp = i.replace('%', '').split()
            if tmp[0] in pred_boxes:
                pred_boxes[tmp[0]].append([int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[2])])
            else:
                pred_boxes[tmp[0]] = [[int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[2])]]
        
        recall = 0
        for i in all_truth:
            tmp = i.split()
            if tmp[0] in truth_boxes:
                truth_boxes[tmp[0]].append([int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5])])
            else:
                truth_boxes[tmp[0]] = [[int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5])]]
            recall += 1
        
        # find the true positives at every detection threshold
        curve_x = []
        curve_y = []
        for thresh in range(101):
            true_positive = 0
            precision = 0
            # find all detected bounding boxes above a certain threshold
            thresh_boxes = {} # containing all bounding boxes with probability higher than a threshold
            for i in pred_boxes: # i is the name of the image
                for j in pred_boxes[i]: # j contains info on one of the bounding boxes on i
                    if j[4] >= thresh: # j[4] is the probability
                        if i in thresh_boxes:
                            thresh_boxes[i].append(j)
                        else:
                            thresh_boxes[i] = [j]
                        precision += 1
            # find true positives
            for i in truth_boxes: # i is the name of the image
                for j in truth_boxes[i]: # j is a particular bounding box
                    if i in thresh_boxes:
                        iou = [0]*len(thresh_boxes[i])
                        for k in range(len(thresh_boxes[i])): # thresh_boxes[i][k] is a particular bounding box
                            if is_intersect(j, thresh_boxes[i][k]):
                                iou[k] = overlap(j, thresh_boxes[i][k])/total(j, thresh_boxes[i][k])
                        if max(iou) > 0.5:
                            true_positive += 1
            if (precision != 0):
                curve_y.append(true_positive/precision)
                curve_x.append(true_positive/recall)
        # plot the curve
        pos += 1
        plt.subplot(len(models), len(datasets), pos)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(model.replace("/", "") + " && " + dataset.replace("/", ""))
        plt.plot(curve_x, curve_y, 'ro')

fig.text(0.5, 0.04, 'Recall', ha='center')
fig.text(0.04, 0.5, 'Precision', va='center', rotation='vertical')
plt.show()
