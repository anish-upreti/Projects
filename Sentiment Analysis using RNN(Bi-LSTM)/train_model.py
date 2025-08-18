import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from data_loader import load_and_prepare_data

# Load and clean dataset
df = load_and_prepare_data("data/IMDB Dataset.csv")

# Tokenization parameters
max_words = 10000  # Use top 10k most frequent words
max_len = 200      # Pad/truncate all reviews to 200 tokens

# Convert text to sequences of integers
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"])
sequences = tokenizer.texts_to_sequences(df["review"])
X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
y = np.array(df["label"])

# Save tokenizer for future predictions
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Bi-LSTM model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),   # Embedding layer maps words to vectors
    Bidirectional(LSTM(64, return_sequences=False)),   # Bi-LSTM captures context from both directions
    Dropout(0.5),                                      # Dropout prevents overfitting
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")                     # Sigmoid for binary classification
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=15, batch_size=64)  # You can increase epochs for better accuracy

# Save model
model.save("models/sentiment_model.h5")
print("✅ Model and tokenizer saved successfully!")

# -------------------------------
# Add evaluation metrics here
# -------------------------------
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
print("\nClassification Report:\n", report)
