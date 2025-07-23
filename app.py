import streamlit as st
import pickle

tfidf=pickle.load(open('vectorizer.pkl','rb'))
modle=pickle.load(open("modle.pkl",'rb'))

st.title('Email/SMS spam Classifier')
input=st.text_input('Enter the massage')

#1 preprocessing
#2 vectorize
#3 predict
#4 display