import gensim
import numpy as np
import pickle
import pandas as pd
#import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import sys
import warnings
warnings.filterwarnings("ignore")

        
# model = load_model('/home/abdalla/_Work/flask_work/BiLSTM_Model')
# tokenizer = pd.read_pickle(r'/home/abdalla/_Work/flask_work/Tokenizer.pickle')
model = load_model('BiLSTM_Model')
tokenizer = pd.read_pickle(r'Tokenizer.pickle')


input_length = 60





def predict(x, model=model):
    """
    this function makes prediction on array of sentences using given model.
    Inputs:
        x -> array of 
    """
    
    x = pd.Series(x.split())#.apply(preprocess)
    x = pad_sequences(tokenizer.texts_to_sequences(x), maxlen=input_length)
    
    pred = model.predict(x)
        
    return list(np.where(pred[:,0]>=0.5, 'positive', 'negative'))

# if __name__=='__main__':
#     print(predict([sys.argv[i] for i in range(1, len(sys.argv))], model))
    
