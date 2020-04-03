import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json

with open("sarcasm.json", 'r') as f:
	json_data = json.load(f)

sentences = []
labels = []
urls = []

for items in json_data:
	sentences.append(items['headline'])
	labels.append(items['is_sarcastic'])
	sentences.append(items['article_link'])

tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")

print("sentence: ", sentences[2])
print("sequence: ", sequences[2])
print("Padding: ", padded[2])
