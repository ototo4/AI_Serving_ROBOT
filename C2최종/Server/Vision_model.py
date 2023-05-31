import os, platform, sys, time
from datetime import date
print('OS name:', os.name, ', system:', platform.system(), ', release:', platform.release())
print("Anaconda version:")
#get_ipython().system('conda list anaconda')
print("Python version:", sys.version)
print("Python version info: ", sys.version_info)
import cv2
print("OpenCV version:", cv2.__version__)
import numpy as np
print("numpy version:", np.__version__)
import tensorflow as tf
print("Keras, tensorflow version:", tf.keras.__version__, tf.__version__)
from tqdm import tqdm
from collections import defaultdict
from asm_eye_status import * 
import face_recognition
print("Face Recognition version:", face_recognition.__version__)
import imutils
from imutils.video import VideoStream



model = load_model()


def init(video_source):
    
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    # face_cascPath = 'lbpcascade_frontalface.xml'
    
    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='haarcascade_righteye_2splits.xml'
    dataset = 'faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)
   
    # asama: modified to include input stream from a video file
    # run one of the following... input from video file or from integrated camera
    # 1. Either this - Integrated Camera
    source_resolution = (0, 0)

    if video_source == 0:
        print("[LOG] Opening webcam...")
        
        # video_capture = cv2.VideoCapture(0) # if using this one: OpenCV VideoCapture() => very slow 

        print("[LOG] Getting Camera Resolution...")
        # use this cv2.VideoCapture() just to get resolution of camera 
        cam = cv2.VideoCapture(0)
        cam_width = int(cam.get(3))
        cam_height = int(cam.get(4))
        source_resolution = (cam_width, cam_height) 
        print('Camera resolution (width, height) in pixels:', source_resolution)
        cam.release()  # immediately release camera after getting the resolution
        
        # switch to imutils VideoStream() for better bufferred frames' reading from camera
        video_capture = VideoStream(src=0).start() # imutils VideoStream(), much faster

    # 2. Or this one - 2nd: video file (video_source other than 0)
    else:
        current_directory = os.getcwd()
        video_file = './video_face/video.mp4'
        print("[LOG] Opening default video file...", current_directory + video_file)

        # video_capture = cv2.VideoCapture(video_file) # this one using OpenCV VideoCapture() is very slow 

        print("[LOG] Getting Video Resolution...")
        # use this cv2.VideoCapture() just to get resolution of the video file
        cam = cv2.VideoCapture(video_file)
        cam_width = int(cam.get(3))
        cam_height = int(cam.get(4))
        source_resolution = (cam_width, cam_height) 
        print('Video resolution (width, height) in pixels:', source_resolution)
        cam.release()  # immediately release camera after getting the resolution
        
        video_capture = VideoStream(src=video_file).start()  # imutils VideoStream(), much faster
        
#         source_resolution = ()
#         print('Video  resolution (width, height) in pixels:', source_resolution)

    print("[LOG] Collecting images...")

    return (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector,             video_capture, source_resolution) 
    

def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False


def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, data, eyes_detected, source_resolution, img_overlay):
        #  ret, frame = video_capture.read() # OpenCV version, very slow

        frame = video_capture.read() # imutils VideoStream version, much faster
 
        # video frame resize        
#         # OpenCV version, very slow
#         if ret == True:
#             frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
#             # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
#         else:
#             print('error reading - camera problem or file error?, exiting...')
#             return None

        # imutils VideoStream version for read buffering, much faster
        '''
        print("here2")
        if frame is None:
            print("here check none")
            print('empty frame detected! - camera closed or end of file?, exiting...')
            return frame
        else:
            print("here ~none")
            frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
            # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.
        
        
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        frame = cv2.flip(frame, 1) # flip horizontal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''
        if frame is None:
            print('empty frame detected! - camera closed or end of file?, exiting...')
            test = None
            return frame, test
        else:
            frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
            # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.

        frame = cv2.flip(frame, 1) # flip horizontal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        test = [] #name labeling
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # for each detected face
        for (x,y,w,h) in faces:
            #print("x, y, w, h", (x, y, w, h))
            # Encode the face into a 128-d embeddings vector
            encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]
            #print("encoding:", encoding)
            #print("data type:", type(data["encodings"]))
            # Compare the vector with all known faces encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            #print("MATCHES:", matches)
            
            # For now we don't know the person name
            name = "Unknown"

            # If there is at least one match:
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)
                print("MAX_NAME", name)
            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]

            eyes = []
            
            # Eyes detection
            # check first if eyes are open (with glasses taking into account)
            open_eyes_glasses = open_eyes_detector.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )


            #잡았다 요놈
            # if open_eyes_glasses detect eyes then they are open 
            #print("OPEN EYES: ", len(open_eyes_glasses))
            
            if len(open_eyes_glasses) >= 0:
                eyes_detected[name]+= '1'
                test.append(name)
                for (ex,ey,ew,eh) in open_eyes_glasses:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    
            # otherwise try detecting eyes using left and right_eye_detector
            # which can detect open and closed eyes                
            else:
                # separate the face into left and right sides
                left_face = frame[y:y+h, x+int(w/2):x+w]
                left_face_gray = gray[y:y+h, x+int(w/2):x+w]

                right_face = frame[y:y+h, x:x+int(w/2)]
                right_face_gray = gray[y:y+h, x:x+int(w/2)]

                # Detect the left eye
                left_eye = left_eye_detector.detectMultiScale(
                    left_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                # Detect the right eye
                right_eye = right_eye_detector.detectMultiScale(
                    right_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                eye_status = '1' # we suppose the eyes are open
                
                # For each eye check wether the eye is closed.
                # If one is closed we conclude the eyes are closed
                for (ex,ey,ew,eh) in right_eye:
                    color = (0,255,0)
                    pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)

                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                for (ex,ey,ew,eh) in left_eye:
                    color = (0,255,0)
                    pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                eyes_detected[name] += eye_status

            # current date & time 
            c_datetime = str(date.today()) + ' ' + time.strftime("%H:%M:%S")
            width = source_resolution[0]
            height = source_resolution[1]
            x_hdr = int(width * 0.01) # starting bottom_x is 1% of the width on the top left
            y_hdr = int(height * 0.1) # startgin bottom_y is 10% of the height on the top left
            cv2.putText(frame, c_datetime, (x_hdr, y_hdr), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # location to put image overlay "blinking eye", maintaining 1% distance from top right x & y
            # blinking eyes image resolution to be displayed is 100x40 (width x height)
            x2_hdr = int(width - (.01 * width) - 100) # starting bottom_x     
            y2_hdr = int(height - (.99 * height)) # starting bottom_y  
            
            # Each time, we check if the person has blinked
            # If yes, we display its name
            if isBlinking(eyes_detected[name],3):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display name
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

                # display image overlay
                alpha = 0.4
                added_image = cv2.addWeighted(frame[y2_hdr:y2_hdr+40,x2_hdr:x2_hdr+100,:],alpha, img_overlay[0:40,0:100,:],1-alpha,0)
                # Change the region with the result
                frame[y2_hdr:y2_hdr+40,x2_hdr:x2_hdr+100] = added_image
       
        return frame, test


# def select_source():
#     valid_selections = ('0', '1')
#     prompt = "Please select source:\n         0: Webcam\n         1: Videofile\n"
#     selection = input(prompt)
#     while not(selection in valid_selections):
#         selection = input(prompt)
#     return selection
