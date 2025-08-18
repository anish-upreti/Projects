import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# Load model and tokenizer
model = load_model('gru_nextword_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

SEQ_LENGTH = 5

def predict_top_k_words(model, tokenizer, text_seq, k=3):
    """
    Predict top-k next words given input sequence.
    """
    text_seq = text_seq.lower().split()
    if len(text_seq) < SEQ_LENGTH:
        text_seq = ['']*(SEQ_LENGTH-len(text_seq)) + text_seq
    text_seq = text_seq[-SEQ_LENGTH:]

    # Convert words to integers
    seq = [tokenizer.word_index.get(w, 0) for w in text_seq]
    seq = np.array([seq])

    # Predict probabilities
    preds = model.predict(seq, verbose=0)[0]
    top_indices = preds.argsort()[-k:][::-1]  # top k indices

    # Map indices to words
    top_words = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                top_words.append(word)
                break
    return top_words

# Streamlit UI
st.title("Next Word Prediction (GRU)")
st.write("Enter a sequence of words and the model will predict the top-3 next words.")

user_input = st.text_input("Type here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some words.")
    else:
        predictions = predict_top_k_words(model, tokenizer, user_input)
        st.write(f"Top-3 predicted words: **{', '.join(predictions)}**")
