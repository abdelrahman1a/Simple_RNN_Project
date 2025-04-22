# 🎬 Sentiment Analysis Web App

A simple deep learning project to classify IMDB movie reviews as **positive** or **negative** using an RNN (Recurrent Neural Network). Built with TensorFlow/Keras and deployed using Streamlit.

🔗 **[Live Demo](https://sentiment-analysis-12.streamlit.app/)**

---

## 📌 About the Project

This project uses Natural Language Processing (NLP) to analyze movie reviews and predict sentiment in real-time.

Users can enter their own movie reviews and the model will determine whether the review is **positive** or **negative**.

---

## ⚙️ Technologies Used

- 🧠 TensorFlow / Keras
- 🗃️ IMDB dataset (from Keras)
- 🔠 Word Embedding + SimpleRNN
- 🧹 Text Preprocessing (Tokenization, Padding)
- 🌐 Streamlit for Web Interface

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
