#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:20:06 2020

@author: craig
Script to set a shortcut key of 'e' while labelling images
using labelImg. The shortcut automatically saves the label
and presses 'd' to move on to the next image, making the process
much more efficient.
"""

import pynput
from pynput.keyboard import Key, Controller

keyboard = Controller()

# The currently active modifiers
current = set()

def execute():
    print ('Keypress: e')
    keyboard.press(Key.ctrl.value)
    keyboard.press('s')   #press ctrl s to save
    keyboard.release('s')
    keyboard.release(Key.ctrl.value)
    
    keyboard.press('d')
    keyboard.release('d')  #press d for next
    
    keyboard.press('w')
    keyboard.release('w')  #presses w to create a new rectangle

def on_press(key):
    if key == pynput.keyboard.KeyCode(char='e'):
        current.add(key)
        execute()

def on_release(key):
    if key == pynput.keyboard.KeyCode(char='e'):
        current.remove(key)

with pynput.keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
