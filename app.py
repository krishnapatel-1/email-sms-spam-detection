import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title('Email/SMS Spam Classifier')
user_input = st.text_area('Enter the message')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button('Predict'):
    if user_input:
        transformed_sms = transform_text(user_input)

        if transformed_sms.strip() == "":
            st.warning("Message doesn't contain useful content to classify.")
        else:
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)

            if result == 1:
                st.header('Spam')
            else:
                st.header("Not Spam")
    else:
        st.warning("Please enter a message.")
