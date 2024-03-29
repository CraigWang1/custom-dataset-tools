# Change Into Directory
First, change into this directory to use the scripts:
```
$ cd ../../path_to/custom-dataset-tools/dataset
```
# COCO_format.py
This is a script to take a directory with images and corresponding xml labels in pascal [labelImg](https://github.com/tzutalin/labelImg) format and format a copy into COCO format. This is useful for taking custom datasets and training machine learning models on them. The script can also resize the images.

Resizing is optional.

```
usage: COCO_format.py [-h] [--image_dir IMAGE_DIR] [--annot_dir ANNOT_DIR]
                      [--save_dir SAVE_DIR] [--ext EXT]
                      [--target_size TARGET_SIZE] [--one_side ONE_SIDE]
                      [--train_test_split TRAIN_TEST_SPLIT] [--random]

Put dataset in COCO format for machine learning training.

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to dataset images.
  --annot_dir ANNOT_DIR
                        Directory to image annotations.
  --save_dir SAVE_DIR   Directory path to save entire COCO formatted dataset.
                        (eg: /home/user).
  --ext EXT             Image files extension to resize.
  --target_size TARGET_SIZE
                        Target size to resize as a tuple of 2 integers.
  --one_side ONE_SIDE   Side (int value) to resize image (eg. 512, 1024x556 =>
                        512x278).
  --train_test_split TRAIN_TEST_SPLIT
                        Portion of images used for training expressed as a
                        decimal (eg. 0.9).
  --random              Whether or not to randomize train and val sets
                        (CAREFUL: if chosen, each time script is called on
                        same dataset, the train and val sets will get mixed
                        up, so val set will be contaminated with images the
                        model already trained on.)
```
A note to make is that the `target_size` parameter is for resizing to a specific size that is specified (eg. "(420,69)"), while `one_side` resizing is to resize one side of the image to be the specified side (eg. 420), and the other side goes with it. One_side resizing preserves the aspect ratio, and you only specify the integer side length (eg. 512).

#### Example usage:
```
$ python3 COCO_format.py \
  --image_dir /home/joe/img_dset \
  --annot_dir /home/joe/img_dset \
  --save_dir /home/joe/dset_COCO \
  --ext JPG \
  --one_side 512 \
  --train_test_split 0.9
```
Output directories:
```
data
└── COCO
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    └── images
        ├── train2017
        │   ├── 0.JPG
        │   ├── 1.JPG
        │   └── 3.JPG
        └── val2017
            └── 2.JPG

```
# YOLO_format.py
This is a script to take a directory with images and corresponding xml labels in pascal [labelImg](https://github.com/tzutalin/labelImg) format and format a copy into YOLO format to train YOLO detectors. The script can also resize the images, though resizing is optional.

```
usage: YOLO_format.py [-h] [--image_dir IMAGE_DIR] [--annot_dir ANNOT_DIR]
                      [--save_dir SAVE_DIR] [--ext EXT]
                      [--target_size TARGET_SIZE] [--one_side ONE_SIDE]
                      [--train_test_split TRAIN_TEST_SPLIT] [--random]

Format images dataset in YOLO format.

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to dataset images.
  --annot_dir ANNOT_DIR
                        Directory to image annotations.
  --save_dir SAVE_DIR   Directory path to save entire YOLO formatted
                        dataset. (eg: /home/user).
  --ext EXT             Image files extension.
  --target_size TARGET_SIZE
                        Target size to resize as a tuple of 2 integers.
  --one_side ONE_SIDE   Side (int value) to resize image (eg. 512, 1024x556 =>
                        512x278).
  --train_test_split TRAIN_TEST_SPLIT
                        Portion of images used for training expressed as a
                        decimal (eg. 0.8).
  --random              Whether or not to randomize train and val sets
                        (CAREFUL: if chosen, each time script is called on
                        same dataset, the train and val sets will get mixed
                        up, so val set will be contaminated with images the
                        model already trained on.)
```
#### Example usage:
```
$ python3 YOLO_format.py \
  --image_dir /home/joe/img_dset \
  --annot_dir /home/joe/img_dset \
  --save_dir /home/joe \
  --ext JPG \
  --train_test_split 0.9
```
Output dataset:
```
/home/joe/
└── data
    ├── obj
    │   ├── 4.JPG
    │   ├── 4.txt
    │   ├── 5.JPG
    │   ├── 5.txt
    │   ├── 6.JPG
    │   ├── 6.txt
    │   ├── 7.JPG
    │   └── 7.txt
    ├── obj.data
    ├── obj.names
    ├── test.txt
    └── train.txt
```

# pascal_format.py
The pascal script is similar to the COCO script, except it formats a copy of a [labelImg](https://github.com/tzutalin/labelImg) annotated dataset (which uses pascal xml format) into Pascal VOC 2007 format.
```
usage: pascal_format.py [-h] [--image_dir IMAGE_DIR] [--annot_dir ANNOT_DIR]
                        [--save_dir SAVE_DIR] [--ext EXT]
                        [--target_size TARGET_SIZE] [--one_side ONE_SIDE]
                        [--train_test_split TRAIN_TEST_SPLIT] [--random]

Format images dataset in PASCAL VOC format.

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to dataset images.
  --annot_dir ANNOT_DIR
                        Directory to image annotations.
  --save_dir SAVE_DIR   Directory path to save entire Pascal VOC formatted
                        dataset. (eg: /home/user).
  --ext EXT             Image files extension.
  --target_size TARGET_SIZE
                        Target size to resize as a tuple of 2 integers.
  --one_side ONE_SIDE   Side (int value) to resize image (eg. 512, 1024x556 =>
                        512x278).
  --train_test_split TRAIN_TEST_SPLIT
                        Portion of images used for training expressed as a
                        decimal (eg. 0.8).
  --random              Whether or not to randomize train and val sets
                        (CAREFUL: if chosen, each time script is called on
                        same dataset, the train and val sets will get mixed
                        up, so val set will be contaminated with images the
                        model already trained on.)        
```
#### Example usage:
```
$ python3 pascal_format.py \
  --image_dir /home/joe/img_dset \
  --annot_dir /home/joe/img_dset \
  --save_dir /home/joe \
  --ext JPG \
  --target_size "(512,420)" \
  --train_test_split 0.9
```
Output directories:
```
data
└── VOCdevkit
    └── VOC2007
        ├── Annotations
        │   ├── 0.xml
        │   ├── 1.xml
        │   ├── 2.xml
        │   └── 3.xml
        ├── ImageSets
        │   └── Main
        │       ├── val.txt
        │       └── trainval.txt
        └── JPEGImages
            ├── 0.JPG
            ├── 1.JPG
            ├── 2.JPG
            └── 3.JPG

```
# renumber_dir.py
renumber_dir.py is a script to renumber a folder of images and/or xml annotations in ascending integer order starting from a specified start number.
```
usage: renumber_dir.py [-h] [--image_dir IMAGE_DIR] [--annot_dir ANNOT_DIR]
                       [--ext EXT] [--start START]

Rename images in directories to integers.

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to dataset images.
  --annot_dir ANNOT_DIR
                        Directory to image annotations.
  --ext EXT             Image files extension.
  --start START         The starting number of renumbered images (eg. start on
                        5, 6, etc.)
```
Note: it will not let you renumber a dir of 20 images that start at 0.jpg, 1.jpg, 2.jpg, etc. starting from 3 for example because the new renumbered images would delete the old ones after renaming itself. To avoid this, you can choose a huge start number that would not accidentally delete anything (eg. 100 in this case) and after that choose 3, which would raise no issues.

#### Example

Before:
```
dset
├── 4.JPG
├── 4.xml
├── 5.JPG
├── 5.xml
├── 6.JPG
├── 6.xml
├── 7.JPG
└── 7.xml
```
Command:
```
$ python3 renumber_dir.py \
  --image_dir /home/joe/dset \
  --annot_dir /home/joe/dset \
  --ext JPG \
  --start 0
```
After:
```
dset
├── 0.JPG
├── 0.xml
├── 1.JPG
├── 1.xml
├── 2.JPG
├── 2.xml
├── 3.JPG
└── 3.xml
```
# resize.py
This is a script that takes an input of a directory (or two split directories, one with images, one with annotations) and writes a copy of the data except resized, using the aforementioned `target_size` or `one_side`.

Note: You can choose to resize a folder of images, images and annotations, or just annotations.

```
usage: resize.py [-h] [--image_dir IMAGE_DIR] [--annot_dir ANNOT_DIR]
                 [--save_dir SAVE_DIR] [--ext EXT] [--target_size TARGET_SIZE]
                 [--one_side ONE_SIDE] [--sub_dirs SUB_DIRS]

Resize directory of images and/or annotations.

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to dataset images.
  --annot_dir ANNOT_DIR
                        Directory to image annotations; optional.
  --save_dir SAVE_DIR   Directory path to save resized images and/or
                        annotations. (eg: /home/user).
  --ext EXT             Image files extension.
  --target_size TARGET_SIZE
                        Target size to resize as a tuple of 2 integers.
  --one_side ONE_SIDE   Side (int value) to resize image (eg. 512, 1024x556 =>
                        512x278).
  --sub_dirs SUB_DIRS   Divide the images/annotations into sub_dirs inside of
                        the save_dir (OPTIONAL).
```
#### Example usage:
```
$ python3 resize.py \
  --image_dir /home/joe/img_dset \
  --annot_dir /home/joe/img_set \
  --save_dir /home/joe/resized_dset \
  --ext JPG \
  --one_side 512
```
New folder:
```
resized_dset
├── 0.JPG
├── 0.xml
├── 1.JPG
├── 1.xml
├── 2.JPG
├── 2.xml
├── 3.JPG
└── 3.xml
```

# extract_sub_dirs.py
This is a script to extract images and/or xml annotations from sub directories within a parent directory to a new folder. You can also choose resizing with `--one_side` or `target_size` resizing (optional).

```
usage: extract_sub_dirs.py [-h] [--parent_dir PARENT_DIR] [--images]
                           [--annots] [--save_dir SAVE_DIR] [--ext EXT]
                           [--target_size TARGET_SIZE] [--one_side ONE_SIDE]

Resize directory of images and/or annotations.

optional arguments:
  -h, --help            show this help message and exit
  --parent_dir PARENT_DIR
                        Parent directory to extract sub directory files from.
  --images              Whether or not to extract images.
  --annots              Whether or not to extract xml annotations.
  --save_dir SAVE_DIR   Directory path to save (optionally) resized images and/or
                        annotations. (eg: /home/user).
  --ext EXT             Image files extension.
  --target_size TARGET_SIZE
                        Target size to resize as a tuple of 2 integers.
  --one_side ONE_SIDE   Side (int value) to resize image (eg. 512, 1024x556 =>
                        512x278).
```
#### Example:
Directory with sub directories:
```
before_dir
├── dir_0
│   ├── 4.JPG
│   └── 4.xml
├── dir_1
│   ├── 5.JPG
│   └── 5.xml
├── dir_2
│   ├── 6.JPG
│   └── 6.xml
└── dir_3
    ├── 7.JPG
    └── 7.xml
```
Command:
```
$ python3 extract_sub_dirs.py --parent_dir /home/joe/before_dir --images --annots --ext JPG --save_dir /home/joe/after_dir

No resizing selected.

Mode: Extracting images and annotations.

Extracting images and corresponding annotations...
100%|████████████████████████████████████████████| 4/4 [00:00<00:00, 197.63it/s
```
After using script:
```
after_dir
├── 4.JPG
├── 4.xml
├── 5.JPG
├── 5.xml
├── 6.JPG
├── 6.xml
├── 7.JPG
└── 7.xml
```

# xml_to_json.py
Credit: https://github.com/Tony607/voc2coco 

This is a script to convert a directory of pascal [labelImg](https://github.com/tzutalin/labelImg) xml annotations to a COCO json format.

```
Convert Pascal VOC annotation to COCO format.

optional arguments:
  -h, --help            show this help message and exit
  --xml_dir XML_DIR     Directory path to xml files.
  --json_file JSON_FILE
                        Output COCO format json file.
```
#### Example usage:
```
$ python3 xml_to_json.py \
  --xml_dir /home/joe/annots \
  --json_file /home/joe/coco_annotation.json
```
New file:
```
/home/joe
├── coco_annotations.json
```


