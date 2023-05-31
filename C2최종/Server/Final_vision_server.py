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
from Vision_model import *

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

make_video(IMAGE_PATH, VIDEO_PATH)


# 모델 학습시킨거 load!
#모델 load
with open('data.pickle','rb') as fr:
    data = pickle.load(fr)


if __name__ == "__main__":
    print("[LOG] Initialization...")

    # input in init(video_source); 0:WebCam, 1:VideoFile
    video_source = 1
    #init
    (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, source_resolution) = init(video_source)

    # overlay image (eye clipart) width: 100, height: 40
    img_overlay = cv2.imread('data/icon_eye_100x40.png')
    
    # Define output filename
    out_dir = './output/'
    if video_source == 0: # camera
        out_filename = out_dir + 'camera_face-blink_detect.mp4'
    else: # video file
        out_filename = out_dir + 'video_face-blink_detect.mp4'
        
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 5
    out = cv2.VideoWriter(out_filename, fourcc, frame_rate, source_resolution)
    
    
    eyes_detected = defaultdict(str)
    imshow_label = "Face Liveness Detector - Blinking Eyes (q-quit, p-pause)"
    print("[LOG] Detecting & Showing Images...")

    frame_count =0
    f_det=[]

    while True:
        
        frame,test= detect_and_display(model, video_capture, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, data, eyes_detected, source_resolution, img_overlay)
        if frame is None:
            print("frame is none")
            break
        
        out.write(frame)
        cv2.imshow(imshow_label, frame)
        
        print(test)
        f_det.append(test)
        
        
        frame_count += 1
        if frame_count == 30:
            frame_count == 0
            break
        
        # asama: modified to include p=pause
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord('q'): # q=quit
            break
        elif key_pressed & 0xFF == ord('p'): # p=pause
            cv2.waitKey(-1)
        
            

    print("[LOG] Writing output file...", out_filename)            
    video_capture.stop()
    out.release()
    cv2.destroyAllWindows()
    
    print("[LOG] All done.")
    #cam 끄기
    #(model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, \
     #     video_capture, source_resolution) = init(video_source)
print("len of f_det: ", len(f_det), f_det)
print("f_det.count([]):", f_det.count([])) 
 
    
if len(f_det) == (f_det.count([])) :
    yes_s = 'pass'
    
else:
    while [] in f_det:
        f_det.remove([])
    yes_s = list(set(sum(f_det, [])))
    if ('A1' in f_det) & ('C3' in f_det):
        if f_det.count('A1') > f_det.count('C3'):
            yes_s.remove('C3')
        else:
            yes_s.remove('A1')


print("인식 좌석", yes_s)


#통신 종료
print("Closed...")
s.close()
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

make_video(IMAGE_PATH, VIDEO_PATH)


# 모델 학습시킨거 load!
#모델 load
with open('data.pickle','rb') as fr:
    data = pickle.load(fr)


if __name__ == "__main__":
    print("[LOG] Initialization...")

    # input in init(video_source); 0:WebCam, 1:VideoFile
    video_source = 1
    #init
    (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture, source_resolution) = init(video_source)

    # overlay image (eye clipart) width: 100, height: 40
    img_overlay = cv2.imread('data/icon_eye_100x40.png')
    
    # Define output filename
    out_dir = './output/'
    if video_source == 0: # camera
        out_filename = out_dir + 'camera_face-blink_detect.mp4'
    else: # video file
        out_filename = out_dir + 'video_face-blink_detect.mp4'
        
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 5
    out = cv2.VideoWriter(out_filename, fourcc, frame_rate, source_resolution)
    
    
    eyes_detected = defaultdict(str)
    imshow_label = "Face Liveness Detector - Blinking Eyes (q-quit, p-pause)"
    print("[LOG] Detecting & Showing Images...")

    frame_count =0
    f_det=[]

    while True:
        
        frame,test= detect_and_display(model, video_capture, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, data, eyes_detected, source_resolution, img_overlay)
        if frame is None:
            print("frame is none")
            break
        
        out.write(frame)
        cv2.imshow(imshow_label, frame)
        
        print(test)
        f_det.append(test)
        
        
        frame_count += 1
        if frame_count == 30:
            frame_count == 0
            break
        
        # asama: modified to include p=pause
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord('q'): # q=quit
            break
        elif key_pressed & 0xFF == ord('p'): # p=pause
            cv2.waitKey(-1)
        
            

    print("[LOG] Writing output file...", out_filename)            
    video_capture.stop()
    out.release()
    cv2.destroyAllWindows()
    
    print("[LOG] All done.")
    #cam 끄기
    #(model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, \
     #     video_capture, source_resolution) = init(video_source)
print("len of f_det: ", len(f_det), f_det)
print("f_det.count([]):", f_det.count([])) 
 
    
if len(f_det) == (f_det.count([])) :
    yes_s = 'pass'
    print("No one!")
else:
    while [] in f_det:
        f_det.remove([])
    yes_s = list(set(sum(f_det, [])))
    if ('A1' in f_det) & ('C3' in f_det):
        if f_det.count('A1') > f_det.count('C3'):
            yes_s.remove('C3')
        else:
            yes_s.remove('A1')
            
    print(("You are....", yes_s))
