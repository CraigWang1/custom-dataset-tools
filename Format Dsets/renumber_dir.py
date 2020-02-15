#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:11:00 2020

@author: craig
Script to rename images in a directory and their corresponding xml files into sequential integer order (eg. 0.png, 0.xml, 1.png, 1.xml, etc.)

**Note: Only works if no files are already in sequential integer order because if they are, then the renaming process will override
some files and result in lost data.

"""

import os
import argparse
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

args = parser.parse_args()

# function to get all image and annotation names
def imgs_and_annots():
    files = os.listdir(args.image_dir)
    files = [os.path.basename(file) for file in files]
    
    imgs = list()   #initialize imgs and xml file lists
    annots = list()
    
    for file in files:
        ext = os.path.splitext(file)[1]  #grabs the extension
        if ext == '.' + args.ext:
            imgs.append(file)  #if it's an image, add it to images
        elif ext == '.xml':
            annots.append(file)  #if its an annotation, add it to annots
    return imgs, annots

# function to renumber all images and annotations in directory
def renumber_files():
    imgs, annots = imgs_and_annots()  #get the image and xml file names
    
    num_imgs = len(imgs)
    num_xmls = len(annots)

    # Print error if there is a different number of images and annotations
    assert num_imgs == num_xmls, 'Number of images should match up with number of annotation xml files.'
    
    print('\nRenumbering images...')
    for i, file in enumerate(imgs):
        old_im = os.path.join(args.image_dir, file)   #rename image
        new_im = os.path.join(args.image_dir, '{}.{}'.format(i, args.ext))
        os.rename(old_im, new_im)
        
        xml = os.path.splitext(file)[0] + '.xml'  #get corresponding xml
        old_xml = os.path.join(args.image_dir, xml)   
        new_xml = os.path.join(args.image_dir, '{}.xml'.format(i))
        os.rename(old_xml, new_xml)   #rename xml

# function to edit the filename portion in the xml file
def edit_xmls():
    print('Renumbering corresponding xml annotations...')
    for file in os.listdir(args.annot_dir):
        if os.path.splitext(file)[1] == '.xml':
            file = os.path.join(args.annot_dir, file)   #gets the xml path
            tree = ET.parse(file)    #read in the xml file
            root = tree.getroot()
            
            img_name = os.path.splitext(os.path.basename(file))[0] + '.png'  #the new filename
            #update all filenames
            for filename in root.iter('filename'):
                new_filename = img_name
                filename.text = new_filename
                
            #write to file
            with open(file, 'a') as f:
                tree.write(file)
    print('\nDone!\n')
    
def renumber_and_edit():
    renumber_files()
    edit_xmls()
    
if __name__ == '__main__':
    renumber_and_edit()
