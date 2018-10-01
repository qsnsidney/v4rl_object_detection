# This script should be placed in the parent folders of all dataset folders
# The darknet folder should also be under this
import os
import subprocess
import numpy as np
import cv2

path = os.getcwd()+"/"
darknet_path = os.getcwd()+"/darknet/"
data_file = "cfg/customize.data" 
cfg_file = "cfg/customize_tiny.cfg"
weight_file = "backup/customize.weights"
thresh = 0.01
# info about class and its corresponding color in the segemeted images
# format: key - image name, value - [[lower], [upper]]
objects = {"window": [[110,150,0], [130,255,255]]}
# key is dataset name, value is a list 
# with first element being the image folder, i.e. where the original images are
# and second being the label folder, i.e. where the segmented images are
datasets = {"etrims-db_v1/": ["images/08_etrims-ds/", "annotations/08_etrims-ds/"]}

for dataset, subfolder in datasets.items():
    image_names = os.listdir(dataset+subfolder[0])
    label_names = os.listdir(dataset+subfolder[1])
    for i in range(len(image_names)):
        image_names[i] = image_names[i].split(".")[0]
    for i in range(len(label_names)):
        label_names[i] = label_names[i].split(".")[0]
    names = set(image_names).intersection(set(label_names))
    all_pred = open(path+dataset+"predictions.txt", 'w')
    all_truth = open(path+dataset+"truth.txt", 'w')

    for name in names:
        # run darknet on the original image to generate predictions.png
        # darknet/src/image.c is modified to output the predicted 
        # bounding box coordinates into coordinates.txt
        img = path+dataset+subfolder[0]+name
	cmd = '''
	cd %s
	./darknet detector test %s %s %s %s -thresh %s
	''' %(darknet_path, data_file, cfg_file, weight_file, img+".jpg", thresh)
        subprocess.check_output(cmd, shell=True)
        # copy from coordinates.txt to prediction.txt
        for line in open(darknet_path+"coordinates.txt", "r"):
            all_pred.write(name+" "+line)

        # get coordinates of the ground-truth bounding boxes
        img_truth = cv2.imread(path+dataset+subfolder[1]+name+".png")
        height, width, channel = img_truth.shape
        # convert RGB to HSV
        hsv = cv2.cvtColor(img_truth, cv2.COLOR_BGR2HSV)
        # apply threshold so only window/blue sections in the segmented image are left
        for object_name, bound in objects.items():
            lower = np.array(bound[0])
            upper = np.array(bound[1])
            mask=cv2.inRange(hsv,lower,upper)
            # find all window/blue sections
            img_truth, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
            # output the bounding boxed associated with the contours
            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                b = [x, x+w, y, y+h]
                all_truth.write(name+" "+object_name+" "+" ".join([str(a) for a in b]) + '\n')

    all_pred.close()
    all_truth.close()
