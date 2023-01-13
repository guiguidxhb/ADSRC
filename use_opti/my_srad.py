# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:21:00 2021

@author: guiguidxhb
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from os import listdir

def Gradient(img):
    """
    Gradients of one image with symmetric boundary conditons
    
    Parameters
    -------
    img ； ndarray
    
    Returns
    ------
    grx : ndarry
        one-order froward  difference in the direction of column(axis = 1)
    gry : ndarry
        one-order froward  difference in the direction of row   (axis = 0)
    glx : ndarry
        one-order backward difference in the direction of column(axis = 1)
    gly : ndarry
        one-order backward difference in the direction of row   (axis = 0)
    grc : ndarry
        self-defined difference function    
    """
    #img(i,j-1)
    img_right = np.roll(img,1,axis = 1)
    img_right[:,0] = img[:,0]
    #img(i,j+1)
    img_left  = np.roll(img,-1,axis = 1)
    img_left[:,-1] = img[:,-1]
    #img(i+1,j)
    img_up = np.roll(img,-1,axis = 0)
    img_up[-1] = img[-1]
    #img(i-1,j)
    img_down = np.roll(img,1,axis = 0)
    img_down[0] = img[0]
    
    #img(i,j+1) - img(i,j)
    grx = img_left - img 
    #img(i+1,j) - img(i,j)
    gry = img_up - img
    #img(i,j)  - img(i,j-1)
    glx = img - img_right 
    #img(i,j)   - img(i-1,j)
    gly = img - img_down   
    return grx,gry,glx,gly

def local_variance(img,window_size):
    h = np.ones((window_size,window_size))
    n = h.sum()
    #print(img[5][5])
    c1 = cv2.filter2D(img**2, -1, h, borderType=cv2.BORDER_REFLECT) / n
    #print(c1[5][5])
    mean = cv2.filter2D(img, -1, h, borderType=cv2.BORDER_REFLECT) /n
    c2 = mean ** 2
    
    vu = c1 - c2 + 1e-06
    return vu

def smooth(src_img,Iterations = 100):
    window_size = 3
    noise_variance = 5
    delta_t = 1.667
    src_img.save('test.png')
    src_img = Image.open('test.png')
    img = np.array(src_img, np.float)
    
    for i in range(0,Iterations): 
        vu = local_variance(img,window_size)
        grx,gry,glx,gly = Gradient(img)
        c = noise_variance**2/ vu
        #print(c)
        # cq(i+1,j)
        cq_up = np.roll(c,-1,axis = 0)
        cq_up[-1] = c[-1]
        # cq(i,j+1)
        cq_left = np.roll(c,-1,axis = 1)
        cq_left[:,-1] = c[:,-1]
        
        Div = cq_up*gry - c*gly + cq_left*grx-c*glx
        #print(Div)
        img = img + 1/4*delta_t*Div
        
    img[img>250] = 250
    img[img<5] = 1
    img = Image.fromarray(img.astype(np.uint8))
    return img

