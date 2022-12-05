import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns





df = pd.read_csv('./all-data.csv',delimiter=',',encoding='latin-1',header=None)
df = df.rename(columns={0:'sentiment',1:'Message'})


#st.bar_chart(data=None, *, x=None, y=None, width=0, height=0, use_container_width=True)


def wordcloud(df, text = 'Message'):
    
    # Join all tweets in one string
    corpus = " ".join(str(review) for review in df['Message'])
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
    st.title("Exploratory Data Analysis")
    option = st.selectbox('Choose the type of wordcloud analysis to be displayed:',('Positive', 'Negative', 'Nuetral'))
    st.write("You selected: {} WordCloud".format(option))

    if (option=='Positive'):
        wordcloud(df.loc[df.sentiment == 'positive'].reset_index())
    elif(option=='Negative'):
        wordcloud(df.loc[df.sentiment == 'negative'].reset_index())
    else:
        wordcloud(df.loc[df.sentiment == 'neutral'].reset_index())



    



    