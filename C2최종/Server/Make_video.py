#!/usr/bin/env python
# coding: utf-8

# # 영상만들기

import re
import os
import numpy as np
import cv2

#IMAGE_PATH: image folder path
#VIDEO_PATH: video saving path
def make_video(IMAGE_PATH, VIDEO_PATH):
    path = IMAGE_PATH

    paths = [os.path.join(path , i ) for i in os.listdir(path) if re.search(".jpg$", i )]
    
    ## 정렬 작업
    store1 = []
    store2 = []
    for i in paths :
        if len(i) == 19 :
            store2.append(i)
        else :
            store1.append(i)

    
    paths = list(np.sort(store1)) + list(np.sort(store2))

    print("counts of images : ", len(paths))


    pathIn= VIDEO_PATH
    pathOut = VIDEO_PATH + '/video.mp4'
    fps = 10

    frame_array = []
    for idx , path in enumerate(paths) : 
        #if (idx % 2 == 0) | (idx % 5 == 0) :
        #  continue

        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
        
    for i in range(1000):
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    
    return pathOut
