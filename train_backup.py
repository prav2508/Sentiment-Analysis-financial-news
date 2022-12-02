import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk.corpus

raw_data = pd.read_csv("./all-data.csv", encoding = "ISO-8859-1",header=None)

raw_data = raw_data.rename(columns={0:'Sentiment',1:'Text'})

def removeUnicode( text ):
    return re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

# apply encoding
raw_data['Sentiment'] = LabelEncoder().fit_transform(raw_data['Sentiment'])

# Text Pre-processing

# 1) text normalization
raw_data['Text'] = [text.lower() for text in raw_data['Text']]

# 2)Removing Unicode Characters
raw_data['Text'] = [removeUnicode(text) for text in raw_data['Text']]

# print('yes' if 'to' in stopwords.words('english') else 'no')
# 3) Remove stop words
raw_data['Text'] = raw_data['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))

