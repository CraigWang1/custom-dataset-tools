#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:04:59 2020

@author: craig
"""

import os, glob, cv2, argparse
import os.path as osp
import xml.etree.ElementTree as ET 
from tqdm import tqdm
from pascal_voc_writer import Writer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize directory of images and/or annotations."
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
        help="Directory path to save entire COCO formatted dataset. (eg: /home/user)",
        default="./",
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
        "--annots_only",
        help="Option to resize only annotations",
        action='store_true'
    )

    args = parser.parse_args()

    # make sure both target_size and one_side are not both selected, but one has to be selected
    assert args.target_size is not None or args.one_side is not None, "Please choose either target_ size resizing or one_side resizing. If you only want to resize annotations, please specify --annots_only."
    if args.target_size is not None and args.one_side is not None:
        raise ValueError("Both target_size and one_side resizing cannot be chosen at the same time.")
    #parse target size input from string to python tuple
    if args.target_size:
        args.target_size = eval(args.target_size)
        msg = "--target_size must be a tuple of 2 integers"
        assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  
    
    return args
    
######### ANNOTATION STUFF
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

def new_xml(xml_file, new_f, save_dir, args):
    #parse the xmlfile
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = get_and_check(root, "size", 1)
    width = int(get_and_check(size, "width", 1).text)   # get original width, height
    height = int(get_and_check(size, "height", 1).text)
    if args.annots_only and args.one_side:
        new_width, new_height = new_dims(width, height, common_size=args.one_side)
    elif args.annots_only and args.target_size:
        new_width, new_height = args.target_size
    else:
        new_width, new_height = im_dims(new_f)   #getting the new dims from image is always more reliable, so get it when you can
    writer = Writer(new_f, new_width, new_height)  #initialize new annotation writer
    for obj in get(root, "object"):                    #for each object
        label = get_and_check(obj, "name", 1).text    #get the label
        bndbox = get_and_check(obj, "bndbox", 1)
        xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
        ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1   #get the original xmin, ymin, xmax, ymax
        xmax = int(get_and_check(bndbox, "xmax", 1).text)
        ymax = int(get_and_check(bndbox, "ymax", 1).text)   
        # resize the bboxes
        if args.target_size or args.one_side:  #correct the coords if we resize the image
            xmin, ymin, xmax, ymax = correct_coords(xmin, ymin, xmax, ymax, width, height, new_width, new_height)
        #make sure the bboxes are all good
        assert xmax > xmin
        assert ymax > ymin
        writer.addObject(label, xmin, ymin, xmax, ymax)   #add this object (box) to our new xml
    
    #saves the new xml in the new save directory
    base = os.path.basename(xml_file)
    writer.save(os.path.join(save_dir, base))
    
#HELPER FUNCTIONS
#helper functino to do the calculations   #original width, original height, new width, new height
def correct_coords(xmin, ymin, xmax, ymax, og_w, og_h, new_w, new_h):
    x_ratio = new_w / og_w     #the x_ratio is the new/old
    y_ratio = new_h / og_h    #the y_ratio is new/old, used to correct bboxes
    #new corrected coordinates
    n_xmin = round(xmin * x_ratio)
    n_ymin = round(ymin * y_ratio)
    n_xmax = round(xmax * x_ratio)
    n_ymax = round(ymax * y_ratio)
    
    return n_xmin, n_ymin, n_xmax, n_ymax
    
############################################## END ANNOTATIONS STUFF

# creates the save dir if it doesn't exist already
def create_dirs(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# main function
def copy(args):
    if args.annots_only:
        fnames = glob.glob(os.path.join(args.annot_dir, "*.xml"))   #gets all xml names in annot_dir
        msg = 'There are no annotations in the annotation directory.'
    else:
        fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir  
        msg = 'There are no images in the image directory.'
    assert len(fnames) > 0, msg
    helper_copy(fnames, args)  #resizes and saves the images
    
# helper function to copy and resize images
def helper_copy(imgs, args):
    if args.annot_dir and args.annots_only:
        print('\nResizing annotations only...')
    elif args.annot_dir:
        print('\nResizing images and corresponding annotations...')
    elif not args.annot_dir:
        print('\nResizing images...')
    for f in tqdm(imgs):
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(args.save_dir, new_fname)   #makes the save fpath
        if not args.annots_only: 
            #resize images
            if args.target_size:       #if target_size
                target_resize(f, new_fpath, args.target_size)
            elif args.one_side:                      #otherwise it's one_size resizing
                one_side_resize(f, new_fpath, args.one_side)
        if args.annot_dir:         #if annots are provided, also resize annotations
            xml_file = base_fname + '.xml'
            xml_file = os.path.join(args.annot_dir, xml_file) #get corresponding xml file
            new_xml(xml_file, new_fpath, args.save_dir, args)   #makes new, resized xml
            
# helper function to resize using target_resize 
# (ex. resize 5000x2500 image to (69,420) can distort shapes)
def target_resize(f, save_path, target_size):
    img = cv2.imread(f)
    img = cv2.resize(img, target_size)
    cv2.imwrite(save_path, img) #read image and resize to specified dimensions

# helper function to resize using one_side_resize 
# (ex. resize 5000x2500 image to (512,256), does not distort shapes)
def one_side_resize(f, save_path, common_size):
    og = cv2.imread(f) #read image
    height, width, _ = og.shape #gets dimensions of the image
    resized_width, resized_height = new_dims(width, height, common_size)
    new = cv2.resize(og, (resized_width, resized_height)) #resize image
    cv2.imwrite(save_path, new)                           #and save

# helper function used for one_side_resize
def new_dims(og_w, og_h, common_size):
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

#get image dimensions
def im_dims(f):
    im = cv2.imread(f)
    height, width, _ = im.shape
    return width, height
    
if __name__ == '__main__':
    args = parse_args()
    create_dirs(args.save_dir)
    copy(args)
    print('') #print empty line to look aesthetic