#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:11:00 2020

@author: craig
Script to rename images in a directory and their corresponding xml files.
"""

import os, re, argparse, glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description="Rename images in directories to integers.")
parser.add_argument(
    "--image_dir",
    help="Directory path to dataset images.",
    type=str,
)
parser.add_argument(
    "--annot_dir",
    help="Directory to image annotations.",
    type=str,
)
parser.add_argument(
    "--ext", help="Image files extension.", default="png", type=str
)
parser.add_argument(
    "--start", 
    help="The starting number of renumbered images (eg. start on 5.png, 6.png, etc.)",
    default=0,
    type=int)

args = parser.parse_args()


############################################################HELPER FUNCTIONS

# helper function to sort files by number
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# helper function to check that there is a corresponding xml for each image
def check_matching(imgs, annots):
    for img in imgs:
        corresp_xml = get_base(img) + '.xml'
        corresp_xml = os.path.join(args.annot_dir, corresp_xml)
        assert os.path.isfile(corresp_xml), "The corresponding xml file for {} does not exist at {}.".format(img, corresp_xml)
    for annot in annots:
        corresp_img = get_base(annot) + '.{}'.format(args.ext)
        corresp_img = os.path.join(args.image_dir, corresp_img)
        assert os.path.isfile(corresp_img), "The corresponding image for {} does not exist at {}.".format(annot, corresp_img)


# helper function to sort list of files
def parse_and_sort(files):
    files = [os.path.basename(file) for file in files] #gets the bsename (eg. /home/123.png -> 123.png)
    return sorted(files, key=numericalSort)  #sorts the files

# function to get all image and annotation names
def imgs_and_annots():
    #first grabs all images and annot files and sorts them, to just basename (eg. 123.png, 124.png, etc)
    imgs = parse_and_sort(glob.glob(os.path.join(args.image_dir, '*{}'.format(args.ext))))
    if args.annot_dir:
        annots = parse_and_sort(glob.glob(os.path.join(args.annot_dir, '*.xml')))
    else:
        annots = []
    
    # make sure that number of images matches with number of annotations
    if args.annot_dir:
        num_imgs = len(imgs)
        num_annots = len(annots)
        assert num_imgs == num_annots, 'Number of images should match up with number of annotation xml files.'
        check_matching(imgs, annots)  #make sure each image has corresponding xml
        
    check_dupes(imgs, annots) #make sure that the renumbering process won't accidentally delete images already with that filename/number
    
    return imgs, annots

# helper function to extract basename (eg. 123.png -> 123)
def get_base(file):
    return os.path.splitext(os.path.basename(file))[0]

def is_int(arg):
    try:
        int(arg)
        return True
    except:
        return False

# helper function to make sure that there are no images already with that number
def check_dupes(imgs, annots):
    num_imgs = len(imgs)
    new_range = range(args.start, args.start+num_imgs)  #the new range of image ids that they will have when renumbering
    
    # make sure the existing images aren't in the range that we are going to renumber 
    # because they will get deleted
    for img in imgs:
        base = get_base(img)
        if is_int(base):
            assert int(base) not in new_range, "Your image will get deleted because the renumbered image will replace it. Choose a different start number."
    if args.annot_dir:
        for annot in annots:
            base = get_base(annot)
            if is_int(base):
                assert int(base) not in new_range, "Your annotations will get deleted because the renumbered annotation will replace it. Choose a different start number."


################################################################### MAIN FUNCTIONS
# function to renumber all images and annotations in directory
def renumber_files():
    imgs, annots = imgs_and_annots()  #get the image and xml file names (eg. 123.png, 123.xml)
    
    if args.annot_dir:
        print('\nRenumbering images and xml files...')
    else:
        print('\nRenumbering images...')
        
    for i, file in enumerate(tqdm(imgs)):
        old_im = os.path.join(args.image_dir, file)   #rename image
        new_im = os.path.join(args.image_dir, '{}.{}'.format(i+args.start, args.ext))
        os.rename(old_im, new_im)
        
        if args.annot_dir: #if there's also annotations:
            xml = annots[i]  #get corresponding xml
            old_xml = os.path.join(args.annot_dir, xml)   
            new_xml = os.path.join(args.annot_dir, '{}.xml'.format(i+args.start))
            os.rename(old_xml, new_xml)   #rename xml

# function to edit the filename portion in the xml file
def edit_xmls():
    print('\nEditing corresponding xml annotations...')
    xmls = [xml for xml in os.listdir(args.annot_dir) if os.path.splitext(xml)[1] == '.xml']
    for file in tqdm(xmls):
        file = os.path.join(args.annot_dir, file)   #gets the xml path
        tree = ET.parse(file)    #read in the xml file
        root = tree.getroot()
        
        img_name = os.path.splitext(os.path.basename(file))[0] + '.{}'.format(args.ext)  #the new filename
        #update all filenames
        for filename in root.iter('filename'):
            new_filename = img_name
            filename.text = new_filename
            
        #write to file
        with open(file, 'a') as f:
            tree.write(file)
    
def renumber_and_edit():
    renumber_files()
    if args.annot_dir: #if there's also annotations, edit those
        edit_xmls()
    # print empty line to look aesthetic
    print('')
    
if __name__ == '__main__':
    renumber_and_edit()
