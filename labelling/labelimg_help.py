#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:20:06 2020

@author: craig
Script to set a shortcut key of 'e' while labelling images
using labelImg. The shortcut automatically saves the label (ctrl s), 
moves to the next image ('d'), and opens the rectangle tool ('w'),
making the process much more efficient at one keypress per image.

There is also an 'r' shortcut which zooms in all the way if annotating
small objects (eg. gate poles)
"""

import pynput
from pynput.keyboard import Key

def next_img():
    print ('Keypress: e')
    keyboard.press(Key.ctrl.value)
    keyboard.press('s')   #press ctrl s to save
    keyboard.release('s')
    keyboard.release(Key.ctrl.value)
    
    keyboard.press('d')
    keyboard.release('d')  #press d for next
    
    keyboard.press('w')
    keyboard.release('w')  #presses w to create a new rectangle

def max_zoom():
    """
    Shortcut to zoom in all the way if annotating small objects (press 'r').
    """     
    print("Keypress: r")
    keyboard.press(Key.ctrl.value) #ctrl zoom = zoom in
    mouse.scroll(0, 41)
    keyboard.release(Key.ctrl.value)

def on_press(key):
    if key == pynput.keyboard.KeyCode(char='e'):
        next_img()
    elif key == pynput.keyboard.KeyCode(char='r'):
        max_zoom()
        
def on_release(key):
    if key == pynput.keyboard.KeyCode(char='e'):
        pass

def shortcut():
    # The currently active modifiers
    with pynput.keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    print("\nShortcut activated. \nWorkflow: Draw box, press 'e'. \nIf annotating small images, press 'r' to zoom.\n")
    keyboard = pynput.keyboard.Controller()
    mouse = pynput.mouse.Controller()
    shortcut()
