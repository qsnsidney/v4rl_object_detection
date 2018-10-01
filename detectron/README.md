Facebook detectron uses Python 2.  
For detectron installation, see https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md.  
For detectron usage, see https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md.  
- - -
### Install Caffe2 from source for Python2
Pytorch and caffe2 has been combined to a single repo (https://github.com/pytorch/pytorch). You can install caffe2 only without torch. Official intsallation guide can be found here: https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile.
#### Test Caffe2 installation
python2 -c 'import caffe2; print(caffe2.\_\_file\_\_)'  
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"  
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
#### caffe2_pb2.py
If you encounter errors about "syntax" and "file" being extra arguments, use this file to replace "/usr/local/lib/python2.7/dist-packages/caffe2/proto/caffe2_pb2.py". This seems to be a protobuf version related issue.  
- - -
### Install OpenCV3 from source for Python2
Source files can be found at https://github.com/opencv/opencv and official installation guide can be found at https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html.  
For cmake, use this command instead:  
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D PYTHON_EXECUTABLE=$(which python2) ..
#### Test OpenCV3
python2 -c 'import cv2; print(cv2.\_\_version\_\_)'  
python2 -c 'import cv2; print(cv2.\_\_file\_\_)'  
- - -
### A demo using Jetson TX2 onboard camera
#### onboard_cam.py
This script should be put in ".../detectron/tools". It is adapted from ".../detectron/tools/infer_simple.py" with added Jetson TX2 onboard camera support. Two options, --cfg and --wts, are needed to run successfully and the script has prompt to ask about them.
#### vis_cam.py
This script should be put in ".../detectron/detectron/utils". It is adapted from ".../detectron/detectron/utils/vis.py". The main function being used is "vis_cam_image()".
#### Running the demo
In the detectron main folder, use command:  
(ResNet101 as backbone)  
python2 tools/onboard_cam.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

