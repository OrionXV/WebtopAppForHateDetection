#!pip install torch
#!pip install transformers
#!pip install contractions

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
import logging

from pathlib import Path
import os
import glob
import torch


import re
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy
import contractions

from torch import nn
from transformers import BertModel

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


###################### CLASSES  ######################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        
        self.labels = list(df['label'])
        self.texts =[tokenizer(
            text, 
            padding='max_length', 
            max_length = 256, 
            truncation=True,
            return_tensors="pt") 
        for text in df['text']] #Should actually use cleaned text

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
     
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3, bias = True)
        #self.linear.apply(
        self.relu = nn.ReLU()
        #initial_output_bias = np.array([[3.938462]*768, [15]*768, [5.]*768])
        initial_output_bias = np.array([3.938462, 15, 5.])
        K = torch.Tensor(initial_output_bias)
        #K = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(K,0),0),0)
        #with torch.no_grad():
        #with torch.no_grad():
        #    self.linear.weight = torch.nn.Parameter(K)
        print(self.linear.bias)
        print(self.linear.bias.data)
        with torch.no_grad():
            self.linear.bias.data = self.linear.bias.data + K
        print(self.linear.bias)
        print(self.linear.bias.data)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
  
  
###################### FUNCTIONS  ######################
  

def textCleaner(mystring):
    mystring = " ".join(mystring.split())
    nystring = contractions.fix(mystring)
    #re.sub(r'(!|.)1+', '', text1) 
    return re.sub(r"^\W+", "", mystring)

def textCleaner_adv(mystring):
    newstring = " ".join(x.lower() for x in mystring.split())
    
    newstring = newstring.replace('[^\w\s]','')
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    newstring = emoji_pattern.sub(r'', newstring)
    
    stop = stopwords.words('english')
    newstring = " ".join(x for x in newstring.split() if x not in stop)
    
    nlp = spacy.load('en', disable=['parser', 'ner'])
    doc = nlp(newstring)

    return " ".join([token.lemma_ for token in doc])
  
def evaluater(model, test_data):

    test = Dataset(test_data)
    
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)
    
    Submission = []
    Chance = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():

        for test_input, y in test_dataloader:

            #test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            Submission.append(output.argmax(dim=1))
            Chance.append(torch.max(outputs, 1))
            #acc = (output.argmax(dim=1) == test_label).sum().item()
            #total_acc_test += acc
    test['label'] = Submission
    test['chance'] = Chance
    return test
  
  
###################### MAIN CODE ######################
  
def mainFunc():
    path = Path.cwd().parent
    path = path.parent
    extension = 'csv' #Can be changed to include JSON 
    path = path / 'Model'/ 'input'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    if not result:
        raise Exception("DATA NOT FOUND")
    print(result)
    data_path = path / result[0]
    data = pd.read_csv(data_path, usecols = [ 'id', 'text'])  
    #temp = data.copy()
    data['orignal_text'] = data['text']
    data['text'].apply(lambda x: textCleaner(x))
    
    #temp['clean_text'] = data['text']
    #data = temp.copy()
    #data['clean_text'].apply(lambda x: textCleaner_adv(x))
    #temp['adv_clean_text'] = data['clean_text']
    #data = temp.copy()
    #del temp
    
    newpath = Path('/Model')
    newpath = newpath / 'Model'
    extension = 'pt'
    os.chdir(newpath)
    modelPresent = glob.glob('*.{}'.format(extension))
    
    extension = 'pth'
    os.chdir(newpath)
    modelWeightsPresent = glob.glob('*.{}'.format(extension))
    
    if modelPresent:
        model = torch.load(newpath / modelPresent[0])
    elif modelWeightsPresent:
        model = BertClassifier()
        model.load_state_dict(torch.load(newpath / modelWeightsPresent[0]))    
    else:
        raise Exception("MODEL OR MODEL WEIGHTS ARE NOT PRESENT IN THE DIRECTORY")
    
    data['label'] = np.nan
    
    submission = evaluater(model, data)
    
    submission.to_csv('/output/results.csv')
