#!/bin/bash

"""
@author: ryan yang
"""

file=$(find *.zip | head -n1)
echo currently selected file is $file

echo enter model name:
read model_name

echo enter image extension:
read image_extension

echo train model location from root or yolov8s.pt:
read model_location

echo now training $model_name with yolov8

time=$(date +"%Y_%m_%d_%I_%M_%p")
echo time is now $time
echo unzipping...
mkdir -p /home/avbotz/train/$model_name/$time/
unzip $file -d /home/avbotz/train/$model_name/$time/

cd $model_name/$time

echo converting xml to YOLO
python3 /home/avbotz/train/YOLO_format.py --train_test_split 0.8 \
--image_dir /home/avbotz/train/$model_name/$time/JPEGImages \
--annot_dir /home/avbotz/train/$model_name/$time/Annotations \
--save_dir /home/avbotz/train/$model_name/$time --ext $image_extension \
--model $model_name --time $time

yolo task=detect mode=train \
model=$model_location \
data=data.yaml imgsz=640 plots=True device=0 epochs=200 save_period=2

trap "rm -rf $file" INT
