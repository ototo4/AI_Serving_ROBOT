import os
from gtts import gTTS
import speech_recognition as sr
import playsound
import time

f = open("C:\\Users\\jihyu\\OneDrive\\바탕 화면\\NLP\\nlp_text\\menu_eatornot.txt", "r", encoding='UTF8')
text_to_say = f.read()
text_to_say =  text_to_say #customer님 ~
#read sound
language = "ko"
gtts_object = gTTS(text = text_to_say,
                  lang = language,
                  slow = False)
f.close()
#save sound
Output = 'menu_eatornot3.mp3'
gtts_object.save(Output)
    
#auto play
playsound.playsound(Output)