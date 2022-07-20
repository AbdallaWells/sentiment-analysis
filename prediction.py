import gensim
import numpy as np
import pickle
import pandas as pd
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import sys
import warnings
warnings.filterwarnings("ignore")

        
# model = load_model('/home/abdalla/_Work/flask_work/BiLSTM_Model')
# tokenizer = pd.read_pickle(r'/home/abdalla/_Work/flask_work/Tokenizer.pickle')
model = load_model('BiLSTM_Model')
tokenizer = pd.read_pickle(r'Tokenizer.pickle')

replace = {":D":"smile", ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat',
           
          "isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   
            "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
            "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
            "can't":"cannot","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
            "mustn't":"must not",
          
          "i'm":'i am', ' u ':'you', ' r ':'are', 'some1':'someone', 'yrs':'years', 'hrs':'hours', 'mins':'minuts',
          'secs':'seconds', ' pls ':' please ', 'plz':'please', '2morow':'tomorrow', '2moro':'tomorrow',
          '2day':'today', '4got':'forget', '4gotten':'forget', 'hahah':'haha', 'hahaha':'haha',
          'hahahaha':'haha', "mother's":"mother", "mom's":'mom', "dad's":'dad', "bday":'birthday',
          " lmo ":'lol', 'lolz':'lol', 'rofl':'lol', '<3':' love ', 'thanx':'thanks', 'thnx':'thanks',
          'goood':'good', 'idk': 'i dont know', "i'll":'i will', "you'll":'you will', "we'll":'we will',
          "it'll":'it will', "it's":'it is', "i've":'i have', "you've":"you have", "we've":'we have',
          "they've":'they have', "you'are":'you are', "we'are":'we are', "they'are":'they are', "i'd":'i would',
          "y'all":'you all', "would've":'would have'}
input_length = 60


def preprocess(x):
    """
    this function cleans an input sentece by doing the following:
        - replace urls with word 'URL'
        - replace user tags with word 'USER'
        - remove punctuations
        - lower the whole sentence
        - remove words with less than 2 characters
        - lemmatize words
    Inputs:
        x -> string
    Outputs:
        cleaned sentence -> string
    """
    
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    usr_pattern = '@[^\s]+'
    hashtag_pattern = '#[^\s]+'
    seq_pattern   = r"(.)\1\1+"
    seq_replace_pattern = r"\1"
    punc_pattern = r'[^\w\s]'

    lemmatizer = nltk.WordNetLemmatizer()

    x = re.sub(usr_pattern, 'USER', x)
    x = re.sub(url_pattern, 'URL', x)
    x = re.sub(seq_pattern, seq_replace_pattern, x)    
    
    for key, val in replace.items():
        x = x.replace(key, " "+val+" ")
    
    x = re.sub(punc_pattern, ' ', x)
    x = re.sub('[0-9]+', '', x)
    x = x.lower().split()
    
    return x


def predict(x, model=model):
    """
    this function makes prediction on array of sentences using given model.
    Inputs:
        x -> array of 
    """
    
    x = pd.Series(x).apply(preprocess)
    x = pad_sequences(tokenizer.texts_to_sequences(x), maxlen=input_length)
    
    pred = model.predict(x)
        
    return list(np.where(pred[:,0]>=0.5, 'positive', 'negative'))

# if __name__=='__main__':
#     print(predict([sys.argv[i] for i in range(1, len(sys.argv))], model))
    
