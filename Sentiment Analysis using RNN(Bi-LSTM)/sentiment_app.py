import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("models/sentiment_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # Must match training

st.title("🎭 Sentiment Analysis App")
st.write("Enter a review to see if it is Positive or Negative")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        # Tokenize and pad the input text
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

        # Predict sentiment
        prediction = model.predict(padded)[0][0]

        # Show result with confidence
        if prediction >= 0.5:
            st.success(f"✅ Positive Sentiment ({prediction*100:.2f}%)")
        else:
            st.error(f"❌ Negative Sentiment ({(1-prediction)*100:.2f}%)")
