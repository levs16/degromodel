import argparse
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the model
model = load_model('degromodel.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max_sequence_len
max_sequence_len = 100  # Assuming this was saved or known from the training phase

def generate_text(seed_text, next_words, model, max_sequence_len):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text from a trained model.')
    parser.add_argument('--seed_text', type=str, required=True, help='Initial text to start generation.')
    parser.add_argument('--next_words', type=int, default=50, help='Number of words to generate.')
    
    args = parser.parse_args()
    
    generated_text = generate_text(args.seed_text, args.next_words, model, max_sequence_len)
    print(generated_text)
