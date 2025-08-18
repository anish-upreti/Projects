import matplotlib.pyplot as plt
from preprocess import load_corpus, tokenize_text, create_sequences
from model import build_gru_model
import pickle
import numpy as np

# Parameters
SEQ_LENGTH = 5
EPOCHS = 30
BATCH_SIZE = 128

# Load and preprocess data
text = load_corpus('data/corpus.txt')
tokenizer, total_words = tokenize_text(text)
X, y = create_sequences(tokenizer, text, SEQ_LENGTH)

# Build GRU model
model = build_gru_model(total_words, SEQ_LENGTH)

# Train model and save history
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# Save model and tokenizer
model.save('gru_nextword_model.keras')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Plot Loss & Accuracy
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Compute perplexity
val_loss = model.evaluate(X, y, verbose=0)[0]
perplexity = np.exp(val_loss)
print(f"\nPerplexity: {perplexity:.2f}")
