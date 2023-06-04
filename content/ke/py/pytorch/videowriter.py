# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:20:56 2020

@author: ndb
"""
#import numpy as np
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#fourcc = -1
out = cv2.VideoWriter('output.avi',fourcc,30,(1024,768))
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()



