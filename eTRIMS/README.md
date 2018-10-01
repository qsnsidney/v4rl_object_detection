
### image.c
This is an modified version of darknet/src/image.c that outputs the coordinates of all predicted bounding boxes in an image into "coordinates.txt" in a format "class_name probability(without %) left right top bottom".

### get_bounding_box.py
This script generates both the predicted and ground-truth bounding box info, the former storing in "predictions.txt" with each line in the format "image_name class_name probability left right top bot", and the later in "truth.txt" with each line in the format "image_name class_name left right top bot".

### get_performance.py
This scripts generates the precision-recall curve using "predictions.txt" and "truth.txt" from last step. Currently it only handles single class (i.e. window) but can be adapted to handle more.  

### get_crops.py
Images in the VarCity dataset (i.e. ruemonge) need additional processing as the segemented images have blacked out areas that isn't cropped out from the original ones. This script takes care of it.
