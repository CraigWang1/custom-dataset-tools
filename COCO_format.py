#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:55:44 2020

@author: craig
Script to format an image dataset to COCO format.
"""

import json
import os
import os.path as osp
import glob
import cv2
import xml.etree.ElementTree as ET 
import argparse
import shutil

from pascal_voc_writer import Writer

parser = argparse.ArgumentParser(
    description="Put dataset in COCO format for machine learning training."
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
    type=str,
)
parser.add_argument(
    "--train_test_split",
    help="Portion of images used for training expressed as a decimal (eg. 0.8)",
    default=0.9,
    type=float,
)

args = parser.parse_args()

#set up coco directory paths
coco = osp.join(args.save_dir, 'dataset_COCO/data/COCO')  
coco_imgs_dir = osp.join(coco, 'images')

######################################################### ANNOTATIONS STUFF

START_BOUNDING_BOX_ID = 1

 #If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"gate":1}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    
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

############################################## END ANNOTATIONS STUFF


def create_dirs():
    #coco voc dataset path    
    sub_dirs = ['annotations','images', 'images/train2017', 'images/val2017']  #sub_directories for voc
    
    print('\nCreating COCO directories...\n')
    for sub_dir in sub_dirs:
        path = os.path.join(coco, sub_dir)
        os.makedirs(path)      #create the directories

def resize_and_copy():
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
    msg = "--target-size must be a tuple of 2 integers"
    assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  #see if the inputs are valid
         
    print(
        "{} files to resize from directory `{}` to target size: {}".format(
            len(fnames), args.image_dir, args.target_size
        )
    )
    
    #gets all of the corresponding xml file names (not full path, just name with xml extension)
    xmls = [osp.splitext(osp.basename(f))[0] + '.xml' for f in fnames] 
    xmls = [osp.join(args.annot_dir, xml) for xml in xmls]
    
    #split train and validations images and annotations
    num_imgs = len(fnames)
    ix = int(num_imgs * args.train_test_split)
    
    train_imgs = fnames[:ix]
    train_annots = xmls[:ix]
    val_imgs = fnames[ix:]
    val_annots = xmls[ix:]
    
    helper_copy(train_imgs, val_imgs, resize=True)  #resizes and saves the images
    
    helper_convert(train_annots, val_annots)

def copy():
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
         
    print(
        "{} files to copy from directory `{}`".format(
            len(fnames), args.image_dir
        )
    )
    
    #gets all of the corresponding xml file names (not full path, just name with xml extension)
    xmls = [osp.splitext(osp.basename(f))[0] + '.xml' for f in fnames]  #gets the xml file basename with .xml
    xmls = [osp.join(args.annot_dir, xml) for xml in xmls]   #locates it in the annot_dir
    
    #split train and validations images and annotations
    num_imgs = len(fnames)
    ix = int(num_imgs * args.train_test_split)
    
    train_imgs = fnames[:ix]
    train_annots = xmls[:ix]
    val_imgs = fnames[ix:]
    val_annots = xmls[ix:]
    
    helper_copy(train_imgs, val_imgs, resize=False)  #saves the images
    
    helper_convert(train_annots, val_annots)

def helper_copy(train_imgs, val_imgs, resize=True):
    train_dir = osp.join(coco_imgs_dir, 'train2017')
    val_dir = osp.join(coco_imgs_dir, 'val2017')
    for f in train_imgs:
        print(".", end="", flush=True)
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(train_dir, new_fname)   #makes the save fpath
        if resize:
            img = cv2.imread(f)                 #reads image in
            img = cv2.resize(img, args.target_size) #resizes it if specified
            cv2.imwrite(new_fpath, img)  #saves it to save directory
        else:
            shutil.copyfile(f, new_fpath)
    
    for f in val_imgs:  #repeat process for validation images
        print(".", end="", flush=True)
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(val_dir, new_fname)   #makes the save fpath
        if resize:
            img = cv2.imread(f)                 #reads image in
            img = cv2.resize(img, args.target_size) #resizes it if specified
            cv2.imwrite(new_fpath, img)  #saves it to save directory
        else:
            shutil.copyfile(f, new_fpath)

def helper_convert(train_annots, val_annots):
    train_json = osp.join(coco, 'annotations/instances_train2017.json')
    val_json = osp.join(coco, 'annotations/instances_val2017.json') #set the json fpaths
        #if annotations are provided:
    if args.target_size:
        print('\nAdjusting bbox annotations...')
        temp_train_xmls = osp.join(coco, 'train_temp')
        temp_val_xmls = osp.join(coco, 'val_temp')
        
        os.mkdir(temp_train_xmls)  #make temp files for adjusted xml files
        os.mkdir(temp_val_xmls)
        
        helper_new_xmls(train_annots, save_loc=osp.join(coco, 'train_temp'))
        helper_new_xmls(val_annots, save_loc=osp.join(coco, 'val_temp'))  #creates the new adjusted xml files
        
        train_annots = glob.glob(osp.join(temp_train_xmls, '*'))
        val_annots = glob.glob(osp.join(temp_val_xmls, '*'))  #get all of the adjusted train and val xmls
    
    print('Converting annotations to coco json format...')
    convert(train_annots, train_json)
    convert(val_annots, val_json)   #convert the anntotations xmls into coco json files
    
    if args.target_size:
        shutil.rmtree(temp_train_xmls)
        shutil.rmtree(temp_val_xmls)  #remove the temporary adjusted bbox xmls when done

def helper_new_xmls(xml_paths, save_loc):
    for xml_loc in xml_paths:
        base = osp.splitext(osp.basename(xml_loc))[0] #gets the base image name
        new_save_loc = osp.join(save_loc, base+'.xml')  #configures save location of new xmlfile
        
        base += '.{}'.format(args.ext) #the image name
        og_impath = osp.join(args.image_dir, base)  #finds original image path
        new_impath = osp.join(coco, 'images/train2017', base)   #finds new image path

        new_bbox_xml(xmlfile=xml_loc, 
                     og_impath=og_impath,
                     new_impath=new_impath,
                     save_loc=new_save_loc)      #makes the new xml file

if __name__ == '__main__':
    create_dirs()
    if args.target_size:
        args.target_size = eval(args.target_size)
        resize_and_copy()
    else:
        copy()
    print('\nDone! Successfully created custom dataset in COCO format.')
    print('Dataset is stored at', coco)
    print('')
