#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:31:54 2020

@author: craig

Script to take custom dataset (directory of images + pascal VOC/labelImg annotations)
and format it into YOLO format to train YOLO detectors.
"""

import os, glob, cv2, argparse, random, re, shutil
import xml.etree.ElementTree as ET 
import os.path as osp
from tqdm import tqdm

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
    print("\nGetting categories from xml annotations...")
    classes_names = []
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}

def xml_to_txt(xml_file, txt_file, categories):
    """Function to convert pascal xml annotation to yolo txt format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = get_and_check(root, "size", 1)
    width = int(get_and_check(size, "width", 1).text)
    height = int(get_and_check(size, "height", 1).text)
    #convert to yolo bbox
    with open(txt_file, 'a+') as f:
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) 
            ymin = int(get_and_check(bndbox, "ymin", 1).text) 
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            #adjust to yolo: <class> <x_center> <y_center> <box_width> <box_height> 
            x_center = ((xmin+xmax)/2)/width   #relative to image (between 0-1)
            y_center = ((ymin+ymax)/2)/height
            bbox_w = (xmax-xmin)/width
            bbox_h = (ymax-ymin)/height
            f.write("{} {} {} {} {}\n".format(category_id, x_center, y_center, bbox_w, bbox_h))
        
##################################################################MAIN FUNCTIONS
def create_dirs(save_dir):
    print("\nCreating save directories...") 
    with tqdm(total=1) as pbar:   #make progress bar to look aesthetic on command line
        os.makedirs(os.path.join(save_dir, 'data/obj'))   #create save directory with data
        pbar.update(1)
    
        
def copy():
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
    assert len(fnames) > 0, "No images matching image directory and provided image extension were found. Try changing the image directory of the file extension (EXT, eg. 'jpg')"
    random.shuffle(fnames)  #shuffle them in random order to get balanced train and val sets
        
    #gets all of the corresponding xml file names (not full path, just name with xml extension)
    xmls = [osp.splitext(osp.basename(f))[0] + '.xml' for f in fnames] 
    xmls = [osp.join(args.annot_dir, xml) for xml in xmls]

    categories = get_categories(xmls)   #get categories (classes/labels) for coco format
    new_fpaths = helper_copy(fnames, xmls, categories)  #resizes and saves the images
    
    # write data files
    write_train_test(new_fpaths) #split train and test sets
    write_obj_names(categories)  #write obj.names file
    write_obj_data(num_classes=len(categories))  #write obj.data file

def helper_copy(imgs, xmls, categories):
    img_dir = osp.join(args.save_dir, 'data/obj')
    new_fpaths = []  #return list of new img paths later to write train/test sets
    print('\nCopying over images and corresponding annotations...')
    for f in tqdm(imgs):
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(img_dir, new_fname)   #makes the save fpath
        if args.target_size:
            img = cv2.imread(f)                 #reads image in
            img = cv2.resize(img, args.target_size) #resizes it if specified
            cv2.imwrite(new_fpath, img)  #saves it to save directory
        elif args.one_side:    #if one_side resizing is selected, resize it this way instead
            one_side_resize(f, new_fpath, common_size=args.one_side)   #does not distort shapes
        else:    #if no resizing is selected, then just copy it
            shutil.copyfile(f, new_fpath)
        #now do format corresponding annotation
        xml = base_fname + '.xml'
        xml = osp.join(args.annot_dir, xml)
        txt = base_fname + '.txt'
        txt = osp.join(img_dir, txt)
        xml_to_txt(xml, txt, categories)
        new_fpaths.append(new_fpath)
    return new_fpaths
            
def write_train_test(fnames):
    num_imgs = len(fnames)  #number of images in image directory
    ix = int(args.train_test_split * num_imgs)  #training pics index
    
    # split train and test
    train = sorted(fnames[:ix], key=numericalSort)
    test = sorted(fnames[ix:], key=numericalSort)
    
    print('\nWriting train filenames...')
    with open(os.path.join(args.save_dir, 'data/train.txt'), 'a+') as f:
        for im in tqdm(train):
            f.write(im + '\n')   #writes the train filepaths
    
    print('\nWriting test filenames...')
    with open(os.path.join(args.save_dir, 'data/test.txt'), 'a+') as f:
        for im in tqdm(test):
            f.write(im + '\n')    #writes the test files split, creates new line for next line

def write_obj_names(classes):
    """
    Function to write data/obj.names file with 1 class name on each line.
    Input = dict{name: id}
    """
    print("\nWriting obj.names classes...")
    names = sorted(list(classes.keys()))
    with open(osp.join(args.save_dir, 'data/obj.names'), 'a+') as f:
        for name in tqdm(names):
            f.write(name + '\n')  #write the name + add new line

def write_obj_data(num_classes):
    """Function to write data/obj.data file"""
    print("\nWriting obj.data file...")
    pbar = tqdm(total=1)         #make progress bar for graphical display
    with open(os.path.join(args.save_dir, 'data/obj.data'), 'a+') as f:
        f.write('classes = {}\ntrain = {}\nvalid = {}\nnames = {}\nbackup = backup/'.format(
                num_classes, 
                osp.join(args.save_dir, 'data/train.txt'), 
                osp.join(args.save_dir, 'data/test.txt'),
                osp.join(args.save_dir, 'data/obj.names')
                )
        )
    pbar.update(1)
    pbar.close()
                 
def main():
    """Main function that completes entire operation"""
    create_dirs(args.save_dir)
    copy()
    print('\nSuccessfully formatted to YOLO format!')
    print('Dataset saved at:', os.path.join(args.save_dir, 'data') + '\n')

if __name__ == "__main__":
    main()