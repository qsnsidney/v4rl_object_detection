# This script should be placed in the top folder where all the datasets (e.g. labelmefacade) are. 
# Each dataset should have two subfolders, images (original .jpg) and labels (segmented .png).

import os
import numpy as np
from pycocotools import mask
import cv2
from PIL import Image, ImagePalette # For indexed images
import pycocotools
from pycocotools import mask
import datetime
import json


root_dir = os.getcwd() + "/"
datasets = {"labelmefacade":(128, 0, 0)} #bgr
#datasets = {"labelmefacade":(0, 0, 128), "ruemonge":(255, 0, 0), "etrims": 8, "cvpr": 0}

INFO = {
    "description": "Window Dataset",
    "url": "https://github.com/VIS4ROB-lab/v4rl_object_detection_analysis",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "qsnsidney",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'window',
        'supercategory': 'facade',
    }
]



coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": [],
    "type": "instances"
}


def create_image_info(im_id, im_name, im_size, date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, url=""):
    im_info = {
        "id": im_id, 
        "file_name": im_name.replace(".png", ".jpg"),
        "width": im_size[1], 
        "height": im_size[0], 
        "date_captured": date_captured,
        "license": license_id, 
        "url": url
    }
    return im_info

def indexed_window_to_binary(im, w_id):
    if len(im.shape) > 2:
        mask = (im[..., 0] == w_id).astype('uint8')
    else:
        mask = (im == w_id).astype('uint8') # shape (h, w)
    return mask

def rgb_window_to_binary(im, w_id):
    mask0 = (im[:, :, 0] == w_id[0]).astype('uint8')
    mask1 = (im[:, :, 1] == w_id[1]).astype('uint8')
    mask2 = (im[:, :, 2] == w_id[2]).astype('uint8')
    mask = mask0 * mask1 * mask2
    return mask

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def main():
    for d in datasets:
        print(d)
        label = datasets[d]
        all_images = os.listdir(root_dir+d+"/labels")
        im_id = 1
        seg_id = 1
        for i in all_images:
            if ".png" not in i:
                continue
            im = cv2.imread(root_dir+d+"/labels/"+i)
            im_info = create_image_info(im_id, i, im.shape)
            coco_output["images"].append(im_info)
            if type(label) == int:    
                binary_mask = indexed_window_to_binary(im, label)
            else:
                binary_mask = rgb_window_to_binary(im, label)
            '''
            # Display binary mask
            cv2.imshow("aa", binary_mask*255)
            cv2.waitKey(0)
            '''
            _, contours, hierarchy = cv2.findContours(binary_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            '''
            # Display contour found
            bg = np.zeros(im.shape)
            cv2.drawContours(bg, contours, -1, 255, 1)
            cv2.imshow("aa", bg)
            cv2.waitKey(0)
            '''
            for poly in contours:
                area = cv2.contourArea(poly)
                bbox = cv2.boundingRect(poly)
                poly = poly.flatten().tolist()
                annotation_info = {
                    "id": seg_id, 
                    "image_id": im_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": list(bbox),
                    "segmentation": [poly]
                }
                coco_output["annotations"].append(annotation_info)
                seg_id += 1  
            im_id += 1
    with open('{}labelme.json'.format(root_dir), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


                    
if __name__ == "__main__":
    main()
