# v4rl

Code I wrote during my time at Vision for Robotics Lab at ETH.

## Darknet
Repo for darknet_ros can be found at https://github.com/leggedrobotics/darknet_ros. Original code and documentation can be found at https://github.com/pjreddie/darknet and https://pjreddie.com/darknet/yolo/.
### labelmefacade
This repo contains scripts helpful for training darknet (https://github.com/leggedrobotics/darknet_ros) on objects in from the LabelMeFacade dataset (https://github.com/cvjena/labelmefacade).
Specifically, we are interested in window detection.

### eTRIMS
This repo contains scripts helpful for evaluating training results of darknet on customized objects.
The ground truth is the eTRIMS dataset (http://www.ipb.uni-bonn.de/projects/etrims_db/), since unlike the labelmefacade dataset, eTRIMS is fully-labeled. Other fully-labeled datasets are the Ecole Centrale Paris Facades Database (http://vision.mas.ecp.fr/Personnel/teboul/data.php), and the VarCity Dataset (i.e. CVPR; https://varcity.ethz.ch/3dchallenge/).

Evaluation metrics are the pulled from some of the major vision challenge:
* Intersection of Union (IOU) = Area of Overlap / Area of Union
  * A prediction is correct if IOU is greater than 0.5
* Precision = True Positive / (True Positive + False Positive)
  * Ture positives againt all predicted positives
* Recall = True Positive / (True Positive + False Negative)
  * True positives againt all ground-truth positives
* Average Precision (AP)

Method to determine if a prediction is true positive:

For a ground-truth bounding box, find all IOU values between it and the predicted ones. If there are IOU values > 0.5, count only the highest one as a true positive.

A good description for the metrics can be found here: https://github.com/rafaelpadilla/Object-Detection-Metrics



## Mask RCNN
Original code and documentation of Facebook dectectron can be found at https://github.com/facebookresearch/Detectron.  
Facebook detectron uses Python2, Caffe2 and OpenCV3 with various backbones to choose from. There are several other implementations: 
* Mobile net as backbone, using Tensorflow: https://github.com/GustavZ/Mobile_Mask_RCNN
* FPN and ResNet101 as backbone, using Tensorflow: https://github.com/matterport/Mask_RCNN
However, those implementations run slower and require more resources on Jetson than the original one. 
Note: there seem to be bugs in some Tensorflow versions making the implementations unusable. 1.3 is what is originally used in the repo and 1.8 seem to be working, but nothing in between. 

### detectron
This repo contains info on how to set up dectectron for Jetson TX2 to use the onboard camera.

### Image
This repo contains info on how to convert segmented images (.png files) to COCO format json files.  

### Training
This repo contains info on how to train on custom COCO datasets.



## Notes
This repo contains notes on pytorch, Tensorflow installation on Jetson, and C++ and CMake.

