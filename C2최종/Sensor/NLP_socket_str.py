import os
from gtts import gTTS
import speech_recognition as sr
import time
import playsound

#질문 함수

#customer: 몇번
#text_file: 읽을 파일 이름
#mp3_file: 저장할 소리파일 이름
def Q_TTS(customer, text_file, mp3_file):
    #read text
    read_file = text_file
    f = open(read_file, "r", encoding='UTF8')
    text_to_say = f.read()
    text_to_say = customer + text_to_say #customer님 ~
    #read sound
    language = "ko"
    gtts_object = gTTS(text = text_to_say,
                      lang = language,
                      slow = False)
    f.close()
    #print(text_to_say)
    #save sound
    Output = mp3_file
    gtts_object.save(Output)
    
    #auto play
    playsound.playsound(Output)
    return Output

#customer: 몇번
#string_wrod: 문장
#mp3_file: 저장할 소리파일 이름
def Q_TTS_str(customer, string_word, mp3_file):
    text_to_say = str(customer) + "고객님," + str(string_word)
    language = "ko"
    gtts_object = gTTS(text = text_to_say,
                      lang = language,
                      slow = False)
    Output = mp3_file
    gtts_object.save(Output)
    playsound.playsound(Output)
    return Output


#대답 함수
#customer: 고객
#text_file: speak을 text화
def A_STT(customer,text_file, choice):
    #마이크로 듣기
    f = open(text_file, 'w', encoding='UTF8')
    r = sr.Recognizer() 
    mic = sr.Microphone()
    #대답듣기
    try:
        with mic as source:
            print(customer,"고객님 대답해주세요")
            time.sleep(1)
            audio = r.listen(source)
            print("complete")
        #대답
        gcSTT = r.recognize_google(audio, language ='ko')
        #print(customer,"고객님의 대답은 "+ gcSTT)
        f.write(gcSTT+ '\n')
        f.close()
        
    except sr.UnknownValueError:
        print("고객님 알아듣지 못했어요")
        
    except sr.RequestError as e:
        print("고객님 잘못된 대답입니다. {0}".format(e))
    
    #대답 return
    if choice == 'y':
        gcSTT = choice + gcSTT
    if choice == 'm':
        gcSTT = choice + gcSTT
    return gcSTT