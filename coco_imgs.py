#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:10:41 2020

@author: craig
Script to copy pics from pascal dataset and put it into coco format (see signaturex edet implementation)
"""
import shutil
import os.path as osp


# function to copy trainval pics from pascal dset to coco dset
def train(train_path, im_folder):
    trainim_list = list()  #set up the list of train images
    with open(train_path) as f:
        for img in f:
            img = img.replace('\n', '')   #for every image, take out the \n
            trainim_list.append(img + '.png')   #add it to the to be copied list
    
    for im in trainim_list:
        old_path = osp.join(im_folder, im)
        new_path = osp.join('/home/craig/AVBotz/gate_dataset_COCO/data/COCO/images/train2017', im)  ###configure this
        print("Old path:", old_path)
        print("New_path:", new_path)   #copy the files over to coco dset
        print("")
        shutil.copyfile(old_path, new_path)

# function to copy test pics from pascal dset to coco dset
def val(test_path, im_folder):
    testim_list = list()            #set up the list of test images
    with open(test_path) as f:
        for img in f:
            img = img.replace('\n', '')            #for every image, take out the \n
            testim_list.append(img + '.png')      #add it to the to be copied list
    
    for im in testim_list:
        old_path = osp.join(im_folder, im)
        new_path = osp.join('/home/craig/AVBotz/gate_dataset_COCO/data/COCO/images/train2017', im) ###configure this
        print("Old path:", old_path)
        print("New_path:", new_path)              #copy the files over to coco dset
        print("")
        shutil.copyfile(old_path, new_path)

#function to copy both trainval and test images
def all_imgs(txt_path, im_folder):
    train_path = osp.join(txt_path, 'trainval.txt')
    test_path = osp.join(txt_path, 'test.txt')
    train(train_path, im_folder)
    val(test_path, im_folder)
    print("Successfully copied trainval and test images over to COCO dataset.")
    
train_path = '/home/craig/AVBotz/gate_dataset_pascal/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
test_path = '/home/craig/AVBotz/gate_dataset_pascal/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
im_folder = '/home/craig/AVBotz/gate_dataset_pascal/data/VOCdevkit/VOC2007/JPEGImages'
#train(train_path, im_folder)
#val(test_path, im_folder)    #copy train and test images to the coco dataset

#txt_path = '/home/craig/AVBotz/gate_dataset_pascal/data/VOCdevkit/VOC2007/ImageSets/Main'
#all_imgs(txt_path, im_folder)
        
        
    