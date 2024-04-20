import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the text
with open('training_data.txt', 'r') as file:
    text = file.read()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Convert text to bi-grams (token sequences of length 2)
token_list = tokenizer.texts_to_sequences([text])[0]
bi_grams = [(token_list[i], token_list[i+1]) for i in range(len(token_list) - 1)]

# Create predictors and label for bi-grams
X, labels = zip(*bi_grams)
X, labels = np.array(X), np.array(labels)
Y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define the model
model = Sequential([
    Embedding(total_words, 64, input_length=1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=100, verbose=1)

# Save the trained model
model.save('degrotest.h5')