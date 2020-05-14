# Custom Dataset Tools
With the emergence of object detection in the past decade due to the unrelenting improvements in machine learning, it is more important than ever to be able to simply create and format datasets to efficiently train models. However, this can be a time consuming and confusing task because each implementation might require a different format, which is where this repository comes in.

Custom Dataset Tools contains convenient scripts to help label and format custom datasets to train machine learning object detectors. All of the scripts presume that [LabelImg](https://github.com/tzutalin/labelImg.git) is used to annotate data.

Current tools:
- [LabelImg](https://github.com/tzutalin/labelImg.git) fast shortcut
- Supports the following formats:
  - **COCO**
  - **Pascal VOC** 
  - **YOLO** 
- Other miscellaneous tools

Labelling Shortcut:

<p align="center">
  <img src="https://raw.githubusercontent.com/CraigWang1/custom-dataset-tools/master/images/labelling.gif"/>
</p>

Format custom dataset to COCO:

<p align="center">
  <img src="https://raw.githubusercontent.com/CraigWang1/custom-dataset-tools/master/images/COCO_format.png"/>
</p>

## **Installation**
First, make sure you have [Python 3 installed](https://www.python.org/downloads/) (only for Windows; Mac and Linux already have it preinstalled).

Open terminal and copy-paste (ctrl-shift-v for linux) the following commands:
```
git clone https://github.com/CraigWang1/Convenient-ML-custom-dataset-tools.git
cd custom-dataset-tools
pip install -r requirements.txt
```
To use tools, navigate to the `/dataset` or the `/labelling` directories for more instructions.



I hope these tools help; feel free to let me know should any issues/questions arise.
Original xml_to_json script: https://github.com/Tony607/voc2coco
