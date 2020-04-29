This is a repository filled with tools to help streamline machine learning dataset creation.
In this repository you will find
- scripts to format datasets into PASCAL VOC and COCO format (with resizing image and bbox annotation capabilities)
- a script to convert Pascal (labelImg) annotations to COCO json format
- a labelling shortcut key to help make life easier when annotating images with labelImg
- a simple script to help visualize the predictions of object detection models
- a script to renumber all images and labelImg (Pascal VOC) annotations to sequential integer order (eg. 0.png, 0.xml, 1.png, 1.xml, etc.)

# **INSTALL**
First, make sure you have [Python 3 installed](https://www.python.org/downloads/) (only for Windows; Mac and Linux already have it preinstalled).

## Get Code
Open terminal.

Copy and paste (ctrl-shift-v for linux) the following command:

```
$ git clone https://github.com/CraigWang1/object_detection_tools.git
$ cd object_detection_tools
$ pip install -r requirements.txt
```

## Install Requirements



I hope these tools help; feel free to let me know should any issues/questions arise.
Original xml_to_json script: https://github.com/Tony607/voc2coco
