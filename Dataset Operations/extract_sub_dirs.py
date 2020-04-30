#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:01:18 2020

@author: craig
This is a script to take split sub dirs from labelling images and extract the xmls back into the main dir.
"""

import os, glob, cv2, argparse, re, shutil
import os.path as osp
import xml.etree.ElementTree as ET 
from tqdm import tqdm
from pascal_voc_writer import Writer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize directory of images and/or annotations."
    )
    parser.add_argument(
        "--parent_dir",
        help="Parent directory to extract sub directory files from.",
        type=str
    )
    parser.add_argument(
        "--images",
        help="Whether or not to extract images.",
        action='store_true'
    )
    parser.add_argument(
        "--annots",
        help="Whether or not to extract xml annotations.",
        action='store_true'
    )
    parser.add_argument(
        "--save_dir",
        help="Directory path to save resized images and/or annotations. (eg: /home/user).",
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
    

    args = parser.parse_args()

    assert args.annots or args.images, "Please provide annotations or images to extract."    #make sure there's images or annots
    if args.target_size and args.one_side:  #make sure the two types aren't chosen at same time
        raise ValueError("Both target_size and one_side resizing cannot be chosen at the same time.")
    if not args.target_size and not args.one_side:
        print("\nNo resizing selected.")
    #parse target size input from string to python tuple
    if args.target_size:
        args.target_size = eval(args.target_size)
        msg = "--target_size must be a tuple of 2 integers"
        assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  
    if not args.images and args.annots:
        print("\nMode: Extracting only annotations.")
    elif args.images and not args.annots:
        print("\nMode: Extracting only images.")
    else:
        print("\nMode: Extracting images and annotations.")
    return args

# helper function to sort files by number
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# create save dir if it doesn't exist already
def create_dirs(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

# extract the goods inside the sub dirs
def extract(args):
    # get list of sub dirs in parent dir
    if args.images:
        sub_dirs = sorted([name for name in os.listdir(args.parent_dir) if os.path.isdir(os.path.join(args.parent_dir, name))], key=numericalSort)
        ext = args.ext
    else:
        sub_dirs = sorted([name for name in os.listdir(args.parent_dir) if os.path.isdir(os.path.join(args.parent_dir, name))], key=numericalSort) 
        ext = 'xml'
    # extract file names from sub dirs
    sub_dirs = [osp.join(args.parent_dir, d) for d in sub_dirs]
    files = get_files(sub_dirs, ext)
    
    # print user friendly info
    if args.annots and not args.images:
        print('\nExtracting annotations only...')
    elif args.annots and args.images:
        print('\nExtracting images and corresponding annotations...')
    else:
        print('\nExtracting images...')
    for f in tqdm(files):
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(args.save_dir, new_fname)   #makes the save fpath
        if args.images: 
            #resize images
            if args.target_size:       #if target_size
                target_resize(f, new_fpath, args.target_size)
            elif args.one_side:                      #otherwise it's one_size resizing
                one_side_resize(f, new_fpath, args.one_side)
            else:
                shutil.copyfile(f, new_fpath)    # if no resize, then just copy it
        if args.annots:         #if annots are provided, also resize annotations
            current_dir = osp.dirname(f)
            xml_file = base_fname + '.xml'
            xml_file = os.path.join(current_dir, xml_file) #get corresponding xml file
            new_xml(xml_file, new_fpath, args.save_dir, args)   #makes new, resized xml

# helper function to grab files from list of dirs
def get_files(dirs, ext):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*.{}'.format(ext))), key=numericalSort))  #add the sub_dirs files
    return files
        
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

def new_xml(xml_file, new_f, save_dir, args):
    #parse the xmlfile
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = get_and_check(root, "size", 1)
    width = int(get_and_check(size, "width", 1).text)   # get original width, height
    height = int(get_and_check(size, "height", 1).text)
    if args.one_side:
        new_width, new_height = new_dims(width, height, common_size=args.one_side)
    elif args.target_size:
        new_width, new_height = args.target_size
    else:
        new_width, new_height = (width, height)
    writer = Writer(new_f, new_width, new_height)  #initialize new annotation writer
    for obj in get(root, "object"):                    #for each object
        label = get_and_check(obj, "name", 1).text    #get the label
        bndbox = get_and_check(obj, "bndbox", 1)
        xmin = int(get_and_check(bndbox, "xmin", 1).text) 
        ymin = int(get_and_check(bndbox, "ymin", 1).text)    #get the original xmin, ymin, xmax, ymax
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

# helper functino to do the calculations   #original width, original height, new width, new height
def correct_coords(xmin, ymin, xmax, ymax, og_w, og_h, new_w, new_h):
    x_ratio = new_w / og_w     #the x_ratio is the new/old
    y_ratio = new_h / og_h    #the y_ratio is new/old, used to correct bboxes
    #new corrected coordinates
    n_xmin = round(xmin * x_ratio)
    n_ymin = round(ymin * y_ratio)
    n_xmax = round(xmax * x_ratio)
    n_ymax = round(ymax * y_ratio)
    
    return n_xmin, n_ymin, n_xmax, n_ymax

if __name__ == '__main__':
    args = parse_args()
    create_dirs(args.save_dir)
    extract(args)
    print('')  #add empty line to look aesthetic
    
    


