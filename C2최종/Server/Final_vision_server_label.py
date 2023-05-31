import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import time
import re
import os
from Make_video import *
from IPython.display import Image, display
import torch
import glob
from matplotlib import pyplot as plt

HOST='141.223.140.70'
PORT=12345

#folder path
IMAGE_PATH = './image_face'
VIDEO_PATH = './video_face'

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data1 = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
count = 0

while True:
    path_of_image = IMAGE_PATH + "/image{}.jpg".format(count)
    count += 1
    while len(data1) < payload_size:
        print("Recv: {}".format(len(data1)))
        data1 += conn.recv(4096)

    print("Done Recv: {}".format(len(data1)))
    packed_msg_size = data1[:payload_size]
    data1 = data1[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data1) < msg_size:
        data1 += conn.recv(4096)
    frame_data = data1[:msg_size]
    data1 = data1[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow',frame)
    time.sleep(3)
    if cv2.waitKey(1) == -1:
        cv2.imwrite(path_of_image, frame)
        break

cv2.destroyAllWindows()

VIDEO_PATH = make_video(IMAGE_PATH, VIDEO_PATH)

##model labeling
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp6/weights/best.pt', force_reload=True)

# Open the webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(VIDEO_PATH)

# Webcam frame settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
label =[]
while cap.isOpened():
    
    # Capture frame, if all went good then 'ret==True'
    ret, frame = cap.read()

    # Make detections
    results = model(frame)
    x = results.pandas().xyxy[0].value_counts('name')
    if x.index.tolist() == ['seat']:
        print('sucess')
    
    # Plot detections
    cv2.imshow('YOLO', np.squeeze(results.render()))
    time.sleep(3)
    # If we press the exit-buttom 'q' we end the webcam caption
    if cv2.waitKey(1) == -1:
        cv2.imwrite(path_of_image, frame)
        break

# Close everything in the end
cap.release()
cv2.destroyAllWindows()

if 'seat'in x:
    print("stop!")
    labeling = 'Stop'
else:
    print("Don't stop, just go")
    labeling = 'No'
