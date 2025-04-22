import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the ImdB DataSet
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key , value in word_index.items()}


# load the pretrained model
model = load_model("RNN_imdb.h5")

# Function to decode Revies
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])


# Function to preprocess input text
def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classifty"):
    preprocessed_input=preprocess_input(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


