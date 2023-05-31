import socket 
import mynetlib
from NLP_socket_str import *
import pickle
HOST = '141.223.140.70'
PORT = 12345



try : 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    
    
    ##if문 넣기(컴퓨터 비전)
    #IF PERSON == TRUE
    print('Connection Clear')
    time.sleep(5)
    Output_arrive = Q_TTS("", ".\\nlp_text\\meal_arrive.txt", ".\\nlp_wav\\meal_arrive.mp3")
    time.sleep(3)
    Output_menu = Q_TTS("", ".\\nlp_text\\menu.txt", ".\\nlp_wav\\menu.mp3") #메뉴 읽어주기

    customer_list = ['A1', 'A2', 'A3']
    catch = ['eat_answer', 'menu_choice']
    #conn, addr = s.accept()
    count = 0
    for i in customer_list :
        if i == 'A2' or i == 'A3': 
            continue
        print(i,"고객님^^")
        count += 1
        Output_eat_or_not = Q_TTS(i, ".\\nlp_text\\menu_eatornot.txt", ".\\nlp_wav\\menu_eatornot{}.mp3".format(count))#식사 유무 질문
    
        #1) 식사 유무 질문에 대한 대답 듣기
        answer_eat = A_STT(i, ".\\nlp_text\\eat_yes_or_not.txt", 'y') #고객 답변 ex 네, 먹겠습니다
        s.sendall(answer_eat.encode('utf-8'))
        print("send complete!")
        #2) 서버에a서 식사 유무 학습모델에 pred_yes하여 분류하기
        #학습한 것 받음
        #answer은 
        #answer, yes_or_no = mynetlib.my_recv(1024,s).decode('utf-8')

        #네, 아니요 학습하여 새로 나온 답변
        print("답변 받으려고 대기중....")
        s.settimeout(20)
        answer_eat_cust= s.recv(1024).decode('utf-8')
        print("답변 받기 완료...!")
        print(answer_eat_cust[1:])
        
        #식사 유무에 대한 답변
        #어떤 메뉴? or 추가사항 말해줌
        answer_eat = Q_TTS_str(i, answer_eat_cust[1:], ".\\nlp_wav\\you_eat{}.mp3".format(count))
        #2-1) 식사 안하겠다###############
        if answer_eat_cust[0] == 'n':
            continue
        #2-2) 식사 하겠다
        #어떤 메뉴먹는지 대답

        answer_choice = A_STT(i, ".\\nlp_text\\menu_choice.txt", 'm')
        #mynetlib.my_send(str(answer_choice_t), s).encode('utf-8')
        #test_2 = [word.encode('utf-8') for word in answer_choice_t]
        s.sendall(answer_choice.encode('utf-8'))
        s.settimeout(10)
        # wait필요
        #answer = mynetlib.my_recv(1024,s).decode('utf-8')
        answer_choice_pred = s.recv(1024).decode('utf-8') #string받음
        #2-2-1) 답변
        Output_result = Q_TTS_str(i, answer_choice_pred, ".\\nlp_wav\\your_menu_choice{}.mp3".format(count))
    
    #질문 끝~  
    end_sign = 'end'
    s.sendall(end_sign.encode('utf-8'))

    time.sleep(15)
    #conn.send('즐거운 비행 되십시오')    
    
except Exception as e : 
    print("Failed. Error message : ", e)

finally :
    print('Close')
    s.close()