# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:16:18 2019

@author: Rena
"""
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import cv2 

DATADIR= "C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train"
CATEGORIES= ["abnormal", "normal"]

for category in CATEGORIES:
        path= os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            plt.imshow(img_array, cmap="gray")
            plt.show()
            break
        break
    
            
            