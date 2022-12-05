import keras as ks
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import requests as req
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import sys
import pandas as pd

endpointURl = "https://api.marketaux.com/v1/news/all?api_token=T8ucMUqNlc53dSY08tEFpdMZWZQ2jGIKuFjtUg3X"

max_fatures = 500000
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
stop_words = set(stopwords.words('english'))
  

def predict(text):
    preProcessText(text)
    news =[]
    sentiment  = {0: 'positive',1:'neutral',2:'negative'} 
    news.append(text)
    tokenizer = None
    tensor = np.load('tensor.npy')
    model = ks.models.load_model("financial_Pred_Model.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    seq = tokenizer.texts_to_sequences(news)

    padded = pad_sequences(seq, maxlen=tensor.shape[1], dtype='int32', value=0)

    pred = model.predict(padded)

    labels = ['0','1','2']
    return sentiment[int(labels[np.argmax(pred)])]

def preProcessText(text):
    # text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

# print(predict("Software as a Service Market Technologies and Global Markets Report 2022: Focus on Business Intelligence & Analytics, CRM, Content Management, ERP, Finance & Accounting, HRM, Supply Chain Management - ResearchAndMarkets.com"))

def prepDashboard(apikey):
    fnews = []
    pred_Sent = []
   
    
    r=requests.get("https://api.apilayer.com/financelayer/news?limit=100", headers={"apikey":apikey})

    data = r.json()['data']
        
    for item in data:
        fnews.append(item['title'])
        pred_Sent.append(predict(item['title']))
    
    predDataFrame = pd.DataFrame(columns=['Sentiment','Text'])
    predDataFrame.Sentiment = pred_Sent
    predDataFrame.Text = fnews

    predDataFrame.to_csv("dashboard.csv",index=False)


if len(sys.argv) > 1:
        apikey = sys.argv[1]
        prepDashboard(apikey)
else:
    print("Please enter the api-key to update news.")
