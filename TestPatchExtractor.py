# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:43:28 2017

@author: B
"""

import cv2
import os
import numpy as np
patchSize=150
patchNumber=0
for filename in os.listdir("otest"):
    pfilename=filename[:-4]
    page=cv2.imread("otest/"+filename,0)
    lpage=cv2.imread("ltest/"+pfilename+'.bmp',0)
    os.mkdir("potest/"+pfilename)
    os.mkdir("potest/"+pfilename+'/all')
    os.mkdir("potest/"+pfilename+'/main')
    os.mkdir("potest/"+pfilename+'/side')
    os.mkdir("potest/"+pfilename+'/back')
    rows,cols=page.shape
    for x in range(0,rows-patchSize,patchSize):
        for y in range(0,cols-patchSize,patchSize):
            patch=page[x:x+patchSize,y:y+patchSize]
            lpatch=lpage[x:x+patchSize,y:y+patchSize]
            cv2.imwrite("potest/"+pfilename+'/all/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            m=np.sum([lpatch==0])
            s=np.sum([lpatch==128])
            b=np.sum([lpatch==255])
            if m>1000:
                cv2.imwrite("potest/"+pfilename+'/main/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            elif s>1000:
                cv2.imwrite("potest/"+pfilename+'/side/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            else:
                cv2.imwrite("potest/"+pfilename+'/back/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            patchNumber=patchNumber+1
            
            
            
            