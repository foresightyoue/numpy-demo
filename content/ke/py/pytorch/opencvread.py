# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:21:12 2020

@author: ndb
"""


import cv2
img=cv2.imread("EU-niedebo.jpg")
cv2.imshow("before",img)
print("visit img[0,0]=",img[0,0])
print("visit img[0,0,0]=",img[0,0,0])
print("visit img[0,0,1]=",img[0,0,1])
print("visit img[0,0,2]=",img[0,0,2])
print("visit img[50,0]=",img[50,0])
print("visit img[100,0]=",img[100,0])
for i in range(0,50):
    for j in range(0,100):
        for k in range(0,3):
            img[i,j,k]=255
for i in range(50,100):
    for j in range(0,100):
        img[i,j]=[128,128,128]
for i in range(100,150):
    for j in range(0,100):
        img[i,j]=0
cv2.imshow("after",img)
print("after img[0,0]=",img[0,0])
print("after img[0,0,0]=",img[0,0,0])
print("after img[0,0,1]=",img[0,0,1])
print("after img[0,0,2]=",img[0,0,2])
print("after img[500,0]=",img[50,0])
print("after img[100,0]=",img[100,0])
cv2.waitKey()
cv2.destroyAllWindows()
