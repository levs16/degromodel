import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the previously saved model
model = load_model(input("Input name(with ext.): "))

# Load the tokenizer
# You need to save and load your tokenizer for consistent preprocessing
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to generate text
def generate_text(seed_text, next_words=3):
    max_sequence_len = 50  # You should adjust this based on your training data
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predictions = model.predict(token_list, verbose=0)
        predicted = np.argmax(predictions, axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Shell for interacting with the AI
print("AI Text Generator. Type 'quit' to exit.")
while True:
    seed_text = input("Enter seed text: ")
    if seed_text == 'quit':
        break
    print("Generated text:", generate_text(seed_text))