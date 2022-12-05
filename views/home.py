import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from wordcloud import WordCloud

dashData =pd.read_csv("dashboard.csv")

positveData = dashData.loc[dashData['Sentiment'] == "positive"].reset_index()
neutralData = dashData.loc[dashData['Sentiment'] == "neutral"].reset_index()
negativeData = dashData.loc[dashData['Sentiment'] == "negative"].reset_index()
classCount = [positveData.count()['Sentiment'],neutralData.count()['Sentiment'],negativeData.count()['Sentiment']]
# classCount = [20,80,20]
sentiment = classCount.index(max(classCount))

def drawPieChart():
    data =[positveData.count()['Sentiment'], neutralData.count()['Sentiment'], negativeData.count()['Sentiment']]
    labels = ['Positive', 'Neutral', 'Negative']

  

    source = pd.DataFrame({"category": data, "value": labels})

    fig = px.pie(source, values='category', names='value')
    fig.update_traces(hoverinfo='label+percent',marker=dict(colors=["green","darkblue","red"]))
    st.plotly_chart(fig,use_container_width=True)

def drawMeter():
    _value = None
    _color = None
    if sentiment == 0:
        _value = (classCount[sentiment]/(classCount[0]+classCount[1]+classCount[2])) * 100
        _color = "green"
        _sentiment = "Positive"
    elif sentiment == 1:
        _value = (classCount[sentiment]/(classCount[0]+classCount[1]+classCount[2])) * 100
        _color = "darkblue"
        _sentiment = "Neutral"
    else:
        _value = (classCount[sentiment]/(classCount[0]+classCount[1]+classCount[2])) * 100
        _color = "red"
        _sentiment = "Negative"



    fig = go.Figure()

    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    delta = {'reference': 50},
    value = _value,
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': _color},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}},
    title = {'text': "Market Sentiment"},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    st.markdown("<h1 style='text-align: center; color: "+_color+";'>"+_sentiment+"</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig,use_container_width=True)

def wordcloud(df, text = 'Text'):
    
    # Join all tweets in one string
    corpus = " ".join(str(review) for review in df[text])
    print ("There are {len(corpus)} words in the combination of all review.")


    wordcloud = WordCloud(max_font_size=25, 
                          max_words=100,
                          collocations = False,
                          background_color="white").generate(corpus)
    
    

    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot()

def load_view():
    st.title('Dashboard')

    st.markdown("""---""")
    drawMeter()

    drawPieChart()

    option = st.selectbox('Choose the type of wordcloud analysis to be displayed:',('Positive', 'Negative', 'Neutral'))
    #st.write("You selected: {} WordCloud".format(option))

    if (option=='Positive'):
        wordcloud(positveData)
    elif(option=='Negative'):
        wordcloud(negativeData)
    else:
        wordcloud(neutralData)
