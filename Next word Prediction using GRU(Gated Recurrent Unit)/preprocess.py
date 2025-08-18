import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def load_corpus(file_path):
    """Load text from file and lowercase it."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text

def tokenize_text(text):
    """Tokenize text into sequences of integers."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words

def create_sequences(tokenizer, text, seq_length):
    """
    Create input sequences (X) and labels (y) for training.
    seq_length = number of words in each input sequence
    """
    tokens = tokenizer.texts_to_sequences([text])[0]
    sequences = []

    for i in range(seq_length, len(tokens)):
        seq = tokens[i-seq_length:i+1]  # input sequence + next word
        sequences.append(seq)

    sequences = np.array(sequences)
    X = sequences[:, :-1]  # all words except last one
    y = sequences[:, -1]   # last word (next word)
    return X, y
