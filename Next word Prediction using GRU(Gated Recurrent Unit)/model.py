from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

def build_gru_model(vocab_size, seq_length):
    """
    Build GRU model for next word prediction.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=seq_length))
    model.add(GRU(128))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
