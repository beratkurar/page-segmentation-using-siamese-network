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
for filename in os.listdir("otrain"):
    pfilename=filename[:-4]
    page=cv2.imread("otrain/"+filename,0)
    lpage=cv2.imread("ltrain/"+pfilename+'.bmp',0)
    os.mkdir("potrain/"+pfilename)
    os.mkdir("potrain/"+pfilename+'/all')
    os.mkdir("potrain/"+pfilename+'/main')
    os.mkdir("potrain/"+pfilename+'/side')
    os.mkdir("potrain/"+pfilename+'/back')
    rows,cols=page.shape
    for x in range(0,rows-patchSize,patchSize):
        for y in range(0,cols-patchSize,patchSize):
            patch=page[x:x+patchSize,y:y+patchSize]
            lpatch=lpage[x:x+patchSize,y:y+patchSize]
            cv2.imwrite("potrain/"+pfilename+'/all/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            m=np.sum([lpatch==0])
            s=np.sum([lpatch==128])
            b=np.sum([lpatch==255])
            if m>1000:
                cv2.imwrite("potrain/"+pfilename+'/main/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            elif s>1000:
                cv2.imwrite("potrain/"+pfilename+'/side/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            else:
                cv2.imwrite("potrain/"+pfilename+'/back/'+pfilename+"_patch"+str(patchNumber)+".png",patch)
            patchNumber=patchNumber+1
            
            
            
            