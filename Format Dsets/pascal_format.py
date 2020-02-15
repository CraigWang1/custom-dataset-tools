#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:24:55 2020

@author: craig

Script to format a dataset in PASCAL VOC format.
"""

import os
import glob
import cv2
import xml.etree.ElementTree as ET 
import argparse

from pascal_voc_writer import Writer

parser = argparse.ArgumentParser(
    description="Format images dataset in PASCAL VOC format."
)
parser.add_argument(
    "--image_dir",
    help="Directory path to dataset images.",
    type=str,
)
parser.add_argument(
    "--annot_dir",
    help="Directory to image annotations; optional",
    type=str,
)
parser.add_argument(
    "--save_dir",
    help="Directory path to save entire Pascal VOC formatted dataset. (eg: /home/user)",
    default="./",
    type=str,
)
parser.add_argument(
    "--ext", help="Image files extension to resize.", default="jpg", type=str
)
parser.add_argument(
    "--target_size",
    help="Target size to resize as a tuple of 2 integers.",
    default="(512, 512)",
    type=str,
)
parser.add_argument(
    "--train_test_split",
    help="Portion of images used for training expressed as a decimal (eg. 0.8)",
    default=0.9,
    type=float,
)

args = parser.parse_args()

args.target_size = eval(args.target_size)

#MAIN FUNCTIONS
def create_dirs(voc):
    #pascal voc dataset path    
    sub_dirs = ['Annotations','ImageSets/Main','JPEGImages']  #sub_directories for voc
    
    print('\nCreating Pascal directories...\n')
    for sub_dir in sub_dirs:
        path = os.path.join(voc, sub_dir)
        os.makedirs(path)      #create the directories
    
def resize_and_save(voc, fnames):
    msg = "--target-size must be a tuple of 2 integers"
    assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  #see if the inputs are valid
         
    print(
        "{} files to resize from directory `{}` to target size:{}".format(
            len(fnames), args.image_dir, args.target_size
        )
    )
    
    new_img_path = os.path.join(voc, 'JPEGImages')
    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)
        img = cv2.imread(fname)                 #for every file, resize it and copy it    
        img_small = cv2.resize(img, args.target_size)
        new_fname = "{}.{}".format(str(i), args.ext)
        small_fname = os.path.join(new_img_path, new_fname)
        cv2.imwrite(small_fname, img_small)
        
        #if annotations are provided:
        if args.annot_dir:
            base = os.path.basename(fname)    #get the image name by itself, no path
            annot = os.path.splitext(base)[0]+'.xml'  #gets the corresponding xml
            save_loc = os.path.join(voc, 'Annotations/{}.{}'.format(str(i), 'xml'))  #sets save locations
            annot_loc = os.path.join(args.annot_dir, annot)   #fetches annotation location
            new_bbox_xml(annot_loc, fname, small_fname, save_loc)    #makes the new xml file
    
        
    print(
        "\nDone resizing {} files.\nSaved to directory: `{}`".format(
            len(fnames), new_img_path))

def write_train_test(voc, fnames):
    num_imgs = len(fnames)  #number of images in image directory
    
    ix = int(args.train_test_split * num_imgs)  #training pics index
    
    print('\nWriting trainval filenames...')
    with open(os.path.join(voc, 'ImageSets/Main/trainval.txt'), 'a+') as f:
        for i in range(0, ix):
            f.write(str(i) + '\n')   #writes the trainval files
    
    print('Writing test filenames...')
    with open(os.path.join(voc, 'ImageSets/Main/test.txt'), 'a+') as f:
        for i in range(ix, num_imgs):
            f.write(str(i) + '\n')    #writes the test files split, creates new line for next line
    
    print('\nDone!')
    print('Dataset saved at:', os.path.join(voc) + '\n')
    
    
    
    

#HELPER FUNCTIONS
#functions to correct the bboxes for image resizing
def adjust_bboxes(xmlfile, origin_img_path):
    xmin, ymin, xmax, ymax, name = get_bboxes(xmlfile)   #get the bounding box coords
    im = cv2.imread(origin_img_path)
    origin_height = im.shape[0]       #get the image shape
    origin_width = im.shape[1]
    
    x_ratio = args.target_size[0] / origin_width     #the x_ratio is the new/old
    y_ratio = args.target_size[1] / origin_height    #the y_ratio is new/old, used to correct bboxes
                                                #after resizing the images
    new_xmin = correct_coords(xmin, x_ratio)
    new_ymin = correct_coords(ymin, y_ratio)
    new_xmax = correct_coords(xmax, x_ratio)
    new_ymax = correct_coords(ymax, y_ratio)
    
    return new_xmin, new_ymin, new_xmax, new_ymax, name
    
#helper functino to do the calculation
def correct_coords(values, ratio):
    new_coords = list()
    for value in values:       #changes the value by the ratio to correct annotations for image resizing
        new_coords.append(round(value * ratio))
    return new_coords
    
#extract bounding box coords from an annotation xml file
def get_bboxes(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    xmin = xml_extract('xmin', root)
    ymin = xml_extract('ymin', root)
    xmax = xml_extract('xmax', root)
    ymax = xml_extract('ymax', root)
    name = xml_extract('name', root)
    return xmin, ymin, xmax, ymax, name

def xml_extract(value, root):   #helper function to extract a piece of data
    values = list()
    for value in root.iter(value):
        try:
            values.append(int(value.text))
        except ValueError:
            values.append(value.text)   #for the labelname, as you can't turn a word into an int
    return values

def new_bbox_xml(xmlfile, og_impath, new_impath, save_loc):  #helper function to adjust bounding boxes
    xmin, ymin, xmax, ymax, name = adjust_bboxes(xmlfile, og_impath) #and save annots to a new xml file
    
    for i in range(len(xmin)):
        width = args.target_size[0]
        height = args.target_size[1]

        writer = Writer(new_impath, width, height) # Writer(path, width, height)
        writer.addObject(name[i], xmin[i], ymin[i], xmax[i], ymax[i]) # addObject(name, xmin, ymin, xmax, ymax)
    
    # save(path)
    writer.save(save_loc)

if __name__ == "__main__":
    voc_path = os.path.join(args.save_dir, 'data/VOCdevkit/VOC2007/')  #path of voc format
    create_dirs(voc_path)  #creates the pascal directories
    
    #fetches all the image filenames
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
    
    resize_and_save(voc_path, fnames)  #resizes images and annotations (if provided) and saves
    write_train_test(voc_path, fnames) #writes the trainval.txt and test.txt files
