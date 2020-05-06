#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:24:55 2020

@author: craig

Script to format a dataset in PASCAL VOC format.
"""

import os, glob, cv2, argparse, random, re, shutil
import xml.etree.ElementTree as ET 
from tqdm import tqdm
from pascal_voc_writer import Writer

parser = argparse.ArgumentParser(
    description="Format images dataset in PASCAL VOC format."
)
parser.add_argument(
    "--image_dir",
    help="Directory path to dataset images.",
    type=str
)
parser.add_argument(
    "--annot_dir",
    help="Directory to image annotations; optional",
    type=str
)
parser.add_argument(
    "--save_dir",
    help="Directory path to save entire Pascal VOC formatted dataset. (eg: /home/user)",
    type=str
)
parser.add_argument(
    "--ext", help="Image files extension to resize.", default="png", type=str
)
parser.add_argument(
    "--target_size",
    help="Target size to resize as a tuple of 2 integers.",
    type=str
)
parser.add_argument(
    "--one_side",
    help="Side (int value) to resize image (eg. 512, 1024x556 => 512x278).",
    type=int
)
parser.add_argument(
    "--train_test_split",
    help="Portion of images used for training expressed as a decimal (eg. 0.8)",
    default=0.9,
    type=float
)

args = parser.parse_args()
# parse the arguments
if args.target_size and args.one_side:
    raise ValueError("Target size and one side resizing cannot both be chosen at the same time")
if args.target_size:
    args.target_size = eval(args.target_size)
    msg = "--target-size must be a tuple of 2 integers"
    assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  #see if the inputs are valid
       
###################################################HELPER FUNCTIONS
    
# helper function to sort files by number
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
# helper function to resize using one_side_resize 
# (ex. resize 5000x2500 image to (512,256), does not distort shapes)
def one_side_resize(f, save_path, common_size=512):
    og = cv2.imread(f) #read image
    height, width, _ = og.shape #gets dimensions of the image
    resized_width, resized_height = one_side_dims(width, height, common_size)
    new = cv2.resize(og, (resized_width, resized_height)) #resize image
    cv2.imwrite(save_path, new)                           #and save
    return resized_width, resized_height

# helper function to get the image dimensions for one side resizing
def one_side_dims(og_w, og_h, common_size=512):
    #figuring out the new dimensions of the resized image
    #one side has to be the specified one_side / common_side (ex. 512)
    if og_h > og_w:
        scale = common_size / og_h
        resized_height = common_size
        resized_width = round(og_w * scale)
    else:
        scale = common_size / og_w
        resized_height = round(og_h * scale)
        resized_width = common_size
    return resized_width, resized_height
    

#HELPER FUNCTIONS
#functions to correct the bboxes for image resizing
def adjust_bboxes(xmlfile, origin_img_path, new_w, new_h):
    xmin, ymin, xmax, ymax, name = get_bboxes(xmlfile)   #get the og bounding box coords
    im = cv2.imread(origin_img_path)
    og_h = im.shape[0]       #get the image shape
    og_w = im.shape[1]
    
    x_ratio = new_w / og_w     #the x_ratio is the new/old
    y_ratio = new_h / og_h    #the y_ratio is new/old, used to correct bboxes after resizing images
    
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

def new_bbox_xml(xmlfile, og_impath, new_impath, save_loc, new_w, new_h):  #helper function to adjust bounding boxes
    xmin, ymin, xmax, ymax, name = adjust_bboxes(xmlfile, og_impath, new_w, new_h) #and save annots to a new xml file
    
    for i in range(len(xmin)):
        writer = Writer(new_impath, new_w, new_h) # Writer(path, width, height)
        writer.addObject(name[i], xmin[i], ymin[i], xmax[i], ymax[i]) # addObject(name, xmin, ymin, xmax, ymax)
    
    # save(path)
    writer.save(save_loc)

# helper function to get image dimensions of image
def im_dims(fpath):
    im = cv2.imread(fpath)
    h, w = im.shape[:2]
    return w, h  #return the width and height
    
##################################################################MAIN FUNCTIONS
def create_dirs(voc):
    #pascal voc dataset path    
    sub_dirs = ['Annotations','ImageSets/Main','JPEGImages']  #sub_directories for voc
    
    print('\nCreating Pascal directories...\n')
    for sub_dir in sub_dirs:
        path = os.path.join(voc, sub_dir)
        os.makedirs(path)      #create the directories
    
def resize_and_save(voc, fnames):  #fnames are the direct filepath
    print("Copying over images and corresponding annotations...")
    new_img_path = os.path.join(voc, 'JPEGImages')
    for fname in tqdm(fnames):
        new_fp = os.path.join(new_img_path, os.path.basename(fname))  #New file location
        if args.target_size:
            img = cv2.imread(fname)                 #for every file, resize it and copy it    
            img_small = cv2.resize(img, args.target_size)
            cv2.imwrite(new_fp, img_small)
            resized_w, resized_h = args.target_size
        elif args.one_side:
            resized_w, resized_h = one_side_resize(fname, save_path=new_fp, common_size=args.one_side)
        else:
            shutil.copyfile(fname, new_fp)
            resized_w, resized_h = im_dims(fname)
        
        #for annotations:
        base = os.path.basename(fname)    #get the image name by itself, no path
        base_f = os.path.splitext(base)[0]  #gets the base filename (eg. 'yeet' from 'yeet.png')
        annot = base_f + '.xml'  #gets the corresponding xml
        save_loc = os.path.join(voc, 'Annotations/{}.xml'.format(base_f))  #sets save locations
        annot_loc = os.path.join(args.annot_dir, annot)   #fetches annotation location
        new_bbox_xml(annot_loc, fname, new_fp, save_loc, new_w=resized_w, new_h=resized_h)    #makes the new xml file
    
def write_train_test(voc, fnames):
    fnames = [os.path.splitext(os.path.basename(f))[0] for f in fnames]  #gets just the base (eg. '1' from '1.png') for each file
    num_imgs = len(fnames)  #number of images in image directory
    ix = int(args.train_test_split * num_imgs)  #training pics index
    
    # split trainval and test
    trainval = sorted(fnames[:ix], key=numericalSort)
    test = sorted(fnames[ix:], key=numericalSort)
    
    print('\nWriting trainval filenames...')
    with open(os.path.join(voc, 'ImageSets/Main/trainval.txt'), 'a+') as f:
        for im in tqdm(trainval):
            f.write(im + '\n')   #writes the trainval files
    
    print('\nWriting val filenames...')
    with open(os.path.join(voc, 'ImageSets/Main/val.txt'), 'a+') as f:
        for im in tqdm(test):
            f.write(im + '\n')    #writes the test files split, creates new line for next line
    
    print('\nSuccessfully formatted to Pascal VOC format!')
    print('Dataset saved at:', os.path.join(voc) + '\n')


if __name__ == "__main__":
    voc_path = os.path.join(args.save_dir, 'data/VOCdevkit/VOC2007/')  #path of voc format
    create_dirs(voc_path)  #creates the pascal directories
    
    #fetches all the image filenames
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
    random.shuffle(fnames)  #shuffle them in random order to get balanced train and val sets
    
    resize_and_save(voc_path, fnames)  #resizes images and annotations (if provided) and saves
    write_train_test(voc_path, fnames) #writes the trainval.txt and test.txt files
