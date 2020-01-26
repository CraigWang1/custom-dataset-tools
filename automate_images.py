#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:20:06 2020

@author: craig
"""

import os

dirs = os.listdir('/media/craig/Gate/Attempt_2')

#parse directories
dirs.remove('AAAA')
#dirs.remove('.Trash-1000')

#get rid of the zip folders
for i in range(5):
    for di in dirs:
        if '.zip' in di:
            dirs.remove(di)

imgs = list()
usb = '/media/craig/Gate/Attempt_2'
os.chdir(usb)
for di in dirs:
    dir_path = os.path.join(usb, di)
    os.chdir(dir_path)                #changes directory to access the pictures
    
    for sub_dir in os.listdir(dir_path):
        dir_path = os.path.join(dir_path, sub_dir)
        os.chdir(dir_path)      #changes directory again to access the folder where the pictures are
        
        for img in os.listdir(dir_path):
            img_path = os.path.join(os.getcwd(), img)
            
            img_name = os.path.basename(img_path)
            if img_name not in os.listdir('/media/craig/Gate/Attempt_2/AAAA'):    #if the images isn't already added
                save_path = os.path.join('/media/craig/Gate/Attempt_2/AAAA', img_name) #set save path
                print('Image path:', img_path)
                print('Save path:', save_path)
                #os.rename(img_path, save_path)
                print('Successfully moved {} to {}.'.format(img_name, save_path))
                imgs.append(img_name)
        
