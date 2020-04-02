import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequenc import pad_sequences

sentence = [							# vocab corpus
	'I love my cat',
	'i, love my dog',
	'you love my dog!',
	'do you think, my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")			# out of vocab
tokenizer.fit_on_texts(sentence)

word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(sentence)

print(word_index)
print(sequence)

test_data = [
	'i really love my dog!',
	'my dog really love my friends'
]

sequence = tokenizer.texts_to_sequences(test_data)
print(sequence)

padded = pad_sequences(sequence)  	# padding -> padding="post" or maxlen=5 or  truncating="post"
print(padded)
