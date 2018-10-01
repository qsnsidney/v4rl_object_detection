Summary on how to convert segmentation imagse to COCO format.


### Some Concepts
Colormap
* Dimension: # of color x 3 (since colors are represented with 3 channels)

Indexed Image
* vs. RGB Imgae: RGB images have three channels; indexed images have only one channel, with each pixel a number corresponding to a row in the colormap (colormap is not necessary if only interested in class)
* Colormap and palette are similar concepts
* Benefits: less memory

RLE (Run Length Encoding)  
* A data compressing algorithm that replaces repeating values by the number of times they repeat

### COCO Annotation Types
#### Of a single object
Polygon: list of points along the object contour
#### Of crowds
Column-major RLE: column-major means that instead of reading a binary mask array left-to-right along rows, we read them up-to-down along columns

#### COCO json content
info: (dictionary)  
    description  
    url  
    version  
    year  
    contributor  
    data_created  
licenses:(list of dictionaries)  
    url  
    id  
    name  
images (list of dictionaries)  
    license  
    url  
    file_name  
    height  
    width  
    date_captured  
    id  
type:instances  
annotations: (list of dictionaries)  
    segmentaion (nested list)  
    OR  
    segmentation (dictionary)  
        counts  
        size  
    area  
    iscrowd  
    image_id  
    bbox  
    category_id  
    id  
categories: (list of dictionaries)  
    supercategory  
    id  
    name  

#### Process
* Read in indexed image and know how many classes there are
* The indexed image will be converted to a binary one for each class, then the binary image will be encoded to COCO mask
* Save the COCO masks (each image will have a same number of COCO masks as classes)
Refer to this script: https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/cocostuffhelper.py. Top level function is "pngToCocoResult()".  

### Info on our custom datasets
#### LabelMeFacade
Color codes for labels are (in R:G:B):  
various = 0:0:0  
building = 128:0:0  
car = 128:0:128  
door = 128:128:0  
pavement = 128:128:128  
road = 128:64:0  
sky = 0:128:128  
vegetation = 0:128:0  
window = 0:0:128
#### ruemonge
winodw =255:0:0  
#### eTRIMS
0=background   
1=building  
2=car  
3=door  
4=pavement  
5=road  
6=sky  
7=vegetation   
8=window  
#### CVPR
window=0  

