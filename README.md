In this repository you will find scripts to:
- [labelImg](https://github.com/tzutalin/labelImg.git) fast shortcut
- COCO format custom dataset
- Pascal VOC format custom dataset
- YOLO format custom dataset
- Resize directory of images/annotations
- Extract images/annotations from nested sub directories
- Convert Pascal (labelImg) annotations to COCO json
- Renumber directory of images and labelImg (Pascal VOC) annotations to sequential integer order (eg. 0.png, 0.xml, 1.png, 1.xml, etc.)

# **INSTALL**
First, make sure you have [Python 3 installed](https://www.python.org/downloads/) (only for Windows; Mac and Linux already have it preinstalled).

## Get Code
Open terminal and copy-paste (ctrl-shift-v for linux) the following commands:

```
$ git clone https://github.com/CraigWang1/Convenient-ML-custom-dataset-tools.git
$ cd object_detection_tools
$ pip install -r requirements.txt
```




I hope these tools help; feel free to let me know should any issues/questions arise.
Original xml_to_json script: https://github.com/Tony607/voc2coco
