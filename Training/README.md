## Changes needed to train for custom detection
#### custom_e2e_mask_rcnn_R-50-FPN_1x.yaml
This script is a sample configuration file used for both custom training and inference with newly trained weights. The e2e (end-to-end) means no additional proposal files will be needed.

#### .../detectron/detectron/core/config.py
Most of the configuration parameters should be modified with the yaml file, except "DATA_LOADER.MINIBATCH_QUEUE_SIZE". Its default value is 64 but can eat up a lot of RAM space; I used 8 for Jetson TX2.

#### .../detectron/datasets/dataset_catalog.py
Need to add info about new dataset here. Read more about dataset setup at https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data.

#### .../detectron/detectron/dataseets/dummy_datasets.py
Check this file to change class name for detected objects.

#### Commands
python2 tools/train_net.py     --cfg configs/getting_started/custom_e2e_mask_rcnn_R-50-FPN_1x.yaml     OUTPUT_DIR tmp/detectron-output

python2 tools/infer_simple.py --cfg configs/getting_started/custom_e2e_mask_rcnn_R-50-FPN_1x.yaml --output-dir tmp/detectron-visualizations --image-ext jpg --wts tmp/detectron-output/train/window_train/generalized_rcnn/model_iter?.pkl demo

- - -

## Caffe2 Concepts
#### Blobs
As data and derivatives flow through the network in the forward and backward passes, Caffe stores, communicates, and manipulates the information as blobs: the blob is the standard array and unified memory interface for the framework. The detials of blob describe how information is stored and communicated in and across layers and nets.  
A blob is a wrapper over the actual data being processed and passed along by Caffe, and also under the hood provides synchronization capability between the CPU and GPU. Mathematically, a blob is an N-dimensional array stored in a C-contiguous fashion.  
Caffe stores and communicates data using blobs. Blobs provide a unified memory interface holding data, e.g. batches of images, model parameters, and derivatives for optimization.  
The conventioanl blob dimensions for batches of image data are number N x channel K x height H x width W.  
For fully-connected networks, use 2D blobs (shape(N, D)).  
Parameter blob dimensions vary according to the type and configurations of the layer. 
* For a convolution layer with 96 filters of 11x11 spatial dimentsion and 3 input channel the blob is 96x3x11x11. 
* For a fully-connected layer with 1000 output channels and 1024 input channels the parameter blob is1024x1000.
A blob stores two chunks of memories, data and diff. The former is the normal data that we pass along, and the latter is the gradient computed by the network.

#### Layers
The layer is the essence of a model and the fundamental unit of computation. Layers convolve, filter, pool, take inner products, apply nonlinearities like rectified-linear and sigmoid and other elementwise transformations, normalize, load data, and compute losses.  
Each layer type defines three critical computations: setup, forward, and backward.

bottom blob --layer--> top blob

#### Nets
The net is a set of layers connected in a computation graph. 
