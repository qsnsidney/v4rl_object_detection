For training on customized dataset for customized images, darknet requires the following files:
* In .../catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config
  * cfg/customized.cfg
  * cfg/customized.data
  * cfg/customized.names
  * weights/pretrained_weights
  * backup/
* In .../labelmefacade
  * .txt files with coordinates info in labels/
  * train.txt: with names of all training images
  * test.txt: with names of all validation images

### Useful commands
* Download pretrained weights for training:
  * wget https://pjreddie.com/media/files/darknet53.conv.74
* Run training
  * ./darknet detector train ../darknet_ros/yolo_network_config/cfg/customize.data ../darknet_ros/yolo_network_config/cfg/customize.cfg ../darknet_ros/yolo_network_config/weights/darknet53.conv.74
      * format: ./darknet detector train [.data file] [.cfg file] [pre-trained weight file]
* Inference
  * ./darknet detector test ../darknet_ros/yolo_network_config/cfg/customize.data ../darknet_ros/yolo_network_config/cfg/customize.cfg ../darknet_ros/yolo_network_config/backup/... /home/nvidia/Downloads/labelmefacade/images/... -thresh 0
      * format: ./darknet detector test [.data file] [.cfg file] [self-trained weights file] [image name] -thresh [threshold value]
      
### process_window.py
Process segmented images with opencv to generate bounding box coordinates in .txt file in a "labels" folder under the top level directory, in the format of <object-class> <x> <y> <width> <height>, where x, y, width, and height are relative to the image's width and height; darknet uses datasets with those .txt format labels for training.

### get_names.py
Generate train.txt and test.txt with images existing in both label/ and images/. 100 of the images are for testing while the rest are for training.
