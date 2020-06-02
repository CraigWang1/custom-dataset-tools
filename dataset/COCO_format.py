#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:55:44 2020

@author: craig
Script to format an image dataset to COCO format.
"""

import json, os, glob, cv2, argparse, shutil, random, re, math
import os.path as osp
import xml.etree.ElementTree as ET 
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Put dataset in COCO format for machine learning training."
)
parser.add_argument(
    "--image_dir",
    help="Directory path to dataset images.",
    type=str
)
parser.add_argument(
    "--annot_dir",
    help="Directory to image annotations.",
    type=str
)
parser.add_argument(
    "--save_dir",
    help="Directory path to save entire COCO formatted dataset. (eg: /home/user).",
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
    "--train_test_split",
    help="Portion of images used for training expressed as a decimal (eg. 0.9).",
    default=0.9,
    type=float
)
parser.add_argument(
    "--random",
    help="Whether or not to randomize train and val sets (CAREFUL: if chosen, each time script is called on same dataset, the train and val sets will get mixed up, so val set will be contaminated with images the model already trained on.",
    action="store_true"
)

args = parser.parse_args()

# make sure only either target resize or one_side resize is chosen
if args.target_size is not None and args.one_side is not None:
    raise ValueError("Both target_size and one_side resizing cannot be chosen at the same time.")
#parse target size input from string to python tuple
if args.target_size:
    args.target_size = eval(args.target_size)
    msg = "--target_size must be a tuple of 2 integers"
    assert isinstance(args.target_size, tuple) and len(args.target_size) == 2, msg  
    
######################################################### ANNOTATIONS STUFF

START_BOUNDING_BOX_ID = 1

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

def convert(xml_files, new_dims, json_file, categories):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    #categories = get_categories(xml_files, mode)   #get categories (classes/labels)
    bnd_id = START_BOUNDING_BOX_ID
    for i, xml_file in enumerate(tqdm(xml_files)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = get_and_check(root, "filename", 1).text
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        #if imgs are resized, then take the new dimensions, otherwise you don't need them
        if args.target_size or args.one_side:
            new_width = new_dims[i][0]
            new_height = new_dims[i][1]
        image = {
            "file_name": filename,  #put new dimensions if resized, otherwise just put the og dimensions
            "height": new_height if args.target_size or args.one_side else height,
            "width": new_width if args.target_size or args.one_side else width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories: #i dont think this will happen because the categories came from all the xml files
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) 
            ymin = int(get_and_check(bndbox, "ymin", 1).text) 
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            # resize the bboxes
            if args.target_size or args.one_side:
                xmin, ymin, xmax, ymax = correct_coords(xmin, ymin, xmax, ymax, width, height, new_width, new_height)
            #make sure the bboxes are all good
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
#helper functino to do correct bbox coords   #original width, original height, new width, new height
def correct_coords(xmin, ymin, xmax, ymax, og_w, og_h, new_w, new_h):
    x_ratio = new_w / og_w     #the x_ratio is the new/old
    y_ratio = new_h / og_h    #the y_ratio is new/old, used to correct bboxes
    #new corrected coordinates
    n_xmin = round(xmin * x_ratio)
    n_ymin = round(ymin * y_ratio)
    n_xmax = round(xmax * x_ratio)
    n_ymax = round(ymax * y_ratio)
    
    return n_xmin, n_ymin, n_xmax, n_ymax

# helper function to resize using one_side_resize 
# (ex. resize 5000x2500 image to (512,256), does not distort shapes)
def one_side_resize(f, save_path, common_size):
    og = cv2.imread(f) #read image
    height, width, _ = og.shape #gets dimensions of the image
    resized_width, resized_height = new_dims(width, height, common_size)
    new = cv2.resize(og, (resized_width, resized_height)) #resize image
    cv2.imwrite(save_path, new)                           #and save
    return resized_width, resized_height

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

# helper function to sort files by number
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def train_test_split(fnames):
    """Helper function to get consistent train and val sets."""
    # calculate number train and number test images
    num_files = len(fnames)  #number of images in image directory
    num_train = int(args.train_test_split * num_files)  #training pics index
    num_test = num_files - num_train
    extract_interval = math.ceil(num_files / num_test)   # (eg. pick every 4th file from the list)
    
    # split train and test         
    # subtract one from the ith file index because lists are 0 indexed so 4th file = list[3]
    test = sorted(fnames[::extract_interval], key=numericalSort)  #extract each test image (eg. each 5th image = test)
    train = sorted([f for f in fnames if f not in test], key=numericalSort)  #the train set is the remaining images not in test set
    
    return train, test  #return train and test sets

def check_corresp(fnames):
    """Helper function to check that all images have corresponding annotations."""
    print("\nMaking sure that each image has a corresponding annotation file...")
    no_annots = []  # list of images without annotations
    for f in tqdm(fnames):
        corresp_xml = osp.splitext(osp.basename(f))[0] + '.xml'  #eg. /home/5.png -> 5.xml
        corresp_xml = osp.join(args.annot_dir, corresp_xml)      #eg. 5.xml -> /annots/5.xml
        if not os.path.exists(corresp_xml):
            no_annots.append(f)
    if no_annots:
        print("") #print empty line, look aesthetic
        for f in no_annots:   #print each file without annot
            print("Error! Image {} does not have an xml annotation.".format(f))
        print("") #print another empty line, look aesthetic
        raise FileNotFoundError("Images do not have corresponding xmls. Annotate all images.")
    

############################################## END ANNOTATIONS STUFF/HELPER FUNCTIONS

def create_dirs():
    #coco dataset path    
    coco = osp.join(args.save_dir, 'data/COCO')
    sub_dirs = ['annotations','images', 'images/train2017', 'images/val2017']  #sub_directories for voc
    
    print('\nCreating COCO directories...')
    for sub_dir in sub_dirs:
        path = os.path.join(coco, sub_dir)
        os.makedirs(path)      #create the directories

def copy():
    fnames = glob.glob(os.path.join(args.image_dir, "*.{}".format(args.ext)))   #gets all file names in img_dir
    assert len(fnames) > 0, "No images matching image directory and provided image extension were found. Try changing the image directory or the file extension (EXT, eg. 'jpg')."
    check_corresp(fnames)  #make sure each image has a corresponding annotation.
    if args.random:
        print("random")
        random.shuffle(fnames)  #shuffle them in random order to get balanced train and val sets
    else:
        fnames.sort(key=numericalSort)  #otherwise sort by number so get consistent sets
    
    #gets all of the corresponding xml file names (not full path, just name with xml extension)
    xmls = [osp.splitext(osp.basename(f))[0] + '.xml' for f in fnames] 
    xmls = [osp.join(args.annot_dir, xml) for xml in xmls]

    #split train and validations images and annotations
    train_imgs, val_imgs = train_test_split(fnames)
    train_annots, val_annots = train_test_split(xmls)
    
    #train_dims = list of image dimensions for each image in the set
    train_dims = helper_copy(train_imgs, mode='train')  #resizes and saves the images
    val_dims = helper_copy(val_imgs, mode='val')  #resizes and saves the images
    
    #convert annotations to coco json format
    categories = get_categories(xmls)   #get categories (classes/labels) for coco format
    helper_convert(train_annots, train_dims, categories, mode='train')  #converts xml annotations to coco json format
    helper_convert(val_annots, val_dims, categories, mode='val')

def helper_copy(imgs, mode='train'):
    coco_imgs_dir = osp.join(args.save_dir, 'data/COCO/images')
    img_dir = osp.join(coco_imgs_dir, '{}2017'.format(mode))
    print('\nCopying over {} images...'.format(mode))
    resized_dims = []  #list of resized dimensions, used later for correcting annotations
    for f in tqdm(imgs):
        base_fname = osp.splitext(osp.basename(f))[0]  #takes the base filename without the extension
        new_fname = "{}.{}".format(base_fname, args.ext)   #puts our new extension on
        new_fpath = osp.join(img_dir, new_fname)   #makes the save fpath
        if args.target_size:
            img = cv2.imread(f)                 #reads image in
            img = cv2.resize(img, args.target_size) #resizes it if specified
            cv2.imwrite(new_fpath, img)  #saves it to save directory
            resized_dims.append((args.target_size[0], args.target_size[1]))
        elif args.one_side:    #if one_side resizing is selected, resize it this way instead
            resized_width, resized_height = one_side_resize(f, new_fpath, common_size=args.one_side)   #does not distort shapes
            resized_dims.append((resized_width, resized_height))
        else:    #if no resizing is selected, then just copy it
            shutil.copyfile(f, new_fpath)
    return resized_dims

def helper_convert(annots, dims, categories, mode='train'):
    #set json filepath
    coco = osp.join(args.save_dir, 'data/COCO')
    json = osp.join(coco, 'annotations/instances_{}2017.json'.format(mode))
    print('\nConverting {} annotations to coco json format...'.format(mode))
    convert(annots, dims, json, categories)

if __name__ == '__main__':
    create_dirs()
    copy()
    coco = osp.join(args.save_dir, 'data/COCO')
    print('\nDone! Successfully created custom dataset in COCO format.')
    print('Dataset is stored at', coco + '\n')
