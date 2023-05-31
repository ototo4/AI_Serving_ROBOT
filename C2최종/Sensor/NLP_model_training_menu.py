#!pip install mxnet
#!pip install gluonnlp pandas tqdm
#!pip install sentencepiece
#!pip install transformers
#!pip install torch

#!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

#from google.colab import drive
#drive.mount('/content/drive')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#import gluonnlp as nlp
import numpy as npW
from tqdm import tqdm, tqdm_notebook


#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()


#!pip install openpyxl


import pandas as pd

chatbot_data = pd.read_excel('/home/piai/test/Project_NLP/Model_training_menu/데이터셋3차.xlsx')


chatbot_data.sample(n=10)

chatbot_data.loc[(chatbot_data['type1'] == "비빔밥"), 'type1'] = 0  
chatbot_data.loc[(chatbot_data['type1'] == "가츠동"), 'type1'] = 1 
chatbot_data.loc[(chatbot_data['type1'] == "양고기스테이크"), 'type1'] = 2  
chatbot_data.loc[(chatbot_data['type1'] == "라면"), 'type1'] = 3 
chatbot_data.loc[(chatbot_data['type1'] == "인기"), 'type1'] = 4 
chatbot_data.loc[(chatbot_data['type1'] == "아무거나"), 'type1'] = 5 


data_list = []
for q, label in zip(chatbot_data['input'], chatbot_data['type1'])  :
    data = []
    data.append(q)
    data.append(str(label))

    data_list.append(data)


from sklearn.model_selection import train_test_split
                                                         
dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)
#print(len(dataset_train))
#print(len(dataset_test))

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# Setting parameters
max_len = 64
batch_size = 10
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,   ##클래스 수 조정##
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    
#BERT 모델 불러오기
model_menu = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model_menu.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model_menu.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

train_dataloader

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict_menu(predict_sentence, menu_c): #menu_c는 메뉴 list

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model_menu.eval()
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)
            out = model_menu(token_ids, valid_length, segment_ids)

            test_eval=[]
            
            logits = out
            logits = logits.detach().cpu().numpy()
            #print(logits)
            if np.argmax(logits) == 0:
                menu_c['k'] -= 1
                test_eval.append("비빔밥은 한식으로 4층을 열어주세요.")
            elif np.argmax(logits) == 1:
                menu_c['j'] -= 1
                test_eval.append("가츠동은 일식으로 3층을 열어주세요.")
            elif np.argmax(logits) == 2:
                menu_c['a'] -= 1
                test_eval.append("양고기스테이크는 양식으로 2층을 열어주세요.")
            elif np.argmax(logits) == 3:
                test_eval.append("라면, 승무원을 불러드리겠습니다.")
            elif np.argmax(logits) == 4: #인기
                random_menu = min(menu_c['k'], menu_c['j'], menu_c['a'])
                #result에 그 key값 넣기
                reverse_dict= dict(map(reversed, menu_c.items()))
                result = reverse_dict[random_menu]
                menu_c[result] -= 1
                if result == 'k':
                    test_eval.append("비빔밥은 한식으로 4층을 열어주세요.")
                elif result == 'j':
                    test_eval.append("가츠동은 일식으로 3층을 열어주세요.")
                elif result == 'a':
                    test_eval.append("양고기스테이크는 양식으로 2층을 열어주세요.")
                        
            elif np.argmax(logits) == 5: #아무거나
                #3가지 메뉴 중 가장 많이 남은 것 줌
                random_menu = max(menu_c['k'], menu_c['j'], menu_c['a'])
                #result에 그 key값 넣기
                reverse_dict= dict(map(reversed, menu_c.items()))
                result = reverse_dict[random_menu]
                menu_c[result] -= 1
                if result == 'k':
                    test_eval.append("비빔밥, 한식은 4층을 열어주세요.")
                elif result == 'j':
                    test_eval.append("가츠동, 일식은 3층을 열어주세요.")
                elif result == 'a':
                    test_eval.append("양고기스테이크, 양식은 2층을 열어주세요.")
              
        
            if not test_eval:
                print("죄송합니다 고객님, 다시 한 번 말씀해주세요")
                return False
            
            result_pred = test_eval[0]
            return result_pred
            
