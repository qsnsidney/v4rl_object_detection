# This script should be placed in the parent folders of all dataset folders
import cv2
import numpy as np
import os

path = os.getcwd()+"/"
images = "ruemonge428/images/"
annotations = "ruemonge428/annotations/"
crops = "ruemonge428/crops/"
image_names = os.listdir(path+images)
label_names = os.listdir(path+annotations)
for i in range(len(image_names)):
    image_names[i] = image_names[i].split(".")[0]
for i in range(len(label_names)):
    label_names[i] = label_names[i].split(".")[0]
names = set(image_names).intersection(set(label_names))

for name in names:
    img = cv2.imread(path+annotations+name+".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    original = cv2.imread(path+images+name+".jpg")
    res = cv2.bitwise_and(original, original, mask = thresh)
    cv2.imwrite(path+crops+name+".jpg", res)
