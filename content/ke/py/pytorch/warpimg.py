# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:00:01 2020

@author: ndb
"""


import cv2
img = cv2.imread("EU-niedebo.jpg")
height,width = img.shape[:2]
M = cv2.getRotationMatrix2D((width/2,height/2),45,0.5)
rotate = cv2.warpAffine(img,M,(width,height))
cv2.imshow("original",img)
cv2.imshow("rotation",rotate)
cv2.waitKey()
cv2.destroyAllWindows()