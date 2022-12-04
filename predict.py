import keras as ks
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import requests as req

endpointURl = "https://api.marketaux.com/v1/news/all?api_token=T8ucMUqNlc53dSY08tEFpdMZWZQ2jGIKuFjtUg3X"

max_fatures = 500000
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

def predict(text):
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

print(predict("Germany stocks higher at close of trade; DAX up 0.27%"))

def updateDashboard():
    return None