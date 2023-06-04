# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:36:05 2020

@author: ndb
"""

import numpy as np
import cv2
cap = cv2.VideoCapture('videotest.MOV',cv2.CAP_DSHOW)
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==False:
        print("open videotest failed")
        break
    else:
        print("open video successfully")
    frame=cv2.Canny(frame,100,200)
    cv2.imshow('frame',frame)
    c = cv2.waitKey(1)
    if c==27:
        break
cap.release()
cv2.destroyAllWindows()