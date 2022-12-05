import streamlit as st
import predict

def load_view():
    st.title("Try the Sentiment Analysis Model")
    text = st.text_input(label= "Sample News")

    
    st.markdown("<h1 style='text-align: center;'> Sentiment </h1>", unsafe_allow_html=True)
    if len(text)>1:
        sentiment = predict.predict(text)
        if sentiment == "positive":
            _color = "green"
            st.markdown("<h1 style='text-align: center; color: "+_color+";'>"+sentiment.upper()+"</h1>", unsafe_allow_html=True)
        elif sentiment == "negative":
            _color = "red" 
            st.markdown("<h1 style='text-align: center; color: "+_color+";'>"+sentiment.upper()+"</h1>", unsafe_allow_html=True)
        else:
            _color = "darkblue" 
            st.markdown("<h1 style='text-align: center; color: "+_color+";'>"+sentiment.upper()+"</h1>", unsafe_allow_html=True)