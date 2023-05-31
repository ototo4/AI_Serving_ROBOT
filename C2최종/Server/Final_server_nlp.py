import socket
import cv2
import numpy as np
from NLP_model_training_menu_one import *
from NLP_model_training_yes_one import *
import time
import mynetlib
import pickle
import select

HOST='141.223.140.70'
PORT=12345

model_menu.load_state_dict(torch.load('./Model_training_menu/model_menu.pth', map_location=device))
model_yes.load_state_dict(torch.load('./Model_training_yes/model_yes.pth', map_location=device))

try :
    #TCP 사용
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    #서버의 아이피와 포트번호 지정
    s.bind((HOST,PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)

    #연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    
    #s.settimeout(5.0)
    #conn,addr=s.accept()
    menu_c = {'k': 110, 'j': 130, 'a': 130}
    s.settimeout(5.0)
    conn,addr=s.accept()
    
    while True:
        #nlp
        #받을 때,
        #audio_sentence = my_recv(1024, conn)
        print("send 도착직전!?")
        s.settimeout(30.0)
        audio_sentence = conn.recv(1024).decode('utf-8')
        print("server receive!!")
        
        if audio_sentence[0] == 'y':
            answer = predict_yes(audio_sentence[1:])
            print(answer)
            print("send 출발!!")
            conn.sendall(answer.encode('utf-8'))
            print("send 완료")
            continue
        
        if audio_sentence[0] == 'm':
            answer = predict_menu(audio_sentence[1:], menu_c)
            print(answer, "menu:", menu_c)
            conn.sendall(answer.encode('utf-8')) #string 보냄
            continue
            
        if audio_sentence == 'end':
            print("해당 고객과의 질문 끝!")
            break
            
    time.sleep(5)
    #conn.send('좋은 시간 되세요')
except Exception as e :
    print('Failed, Error message : ', e)
finally :
    print('Close')
    s.close()
