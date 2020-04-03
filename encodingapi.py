import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

print("word Index: ", word_index)
print("train Sequence: ", sequence)

padded = pad_sequences(sequence) 
print("train Padding: ", padded)

test_data = [
	'i really love my dog!',
	'my dog really love my friends'
]

sequence = tokenizer.texts_to_sequences(test_data)
print("test Sequence: ", sequence)

padded = pad_sequences(sequence)  	# padding -> padding="post" or maxlen=5 or  truncating="post"
print("test Padding: ", padded)

# output:

# word Index:  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 
#				'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

# train Sequence:  [[5, 3, 2, 7], [5, 3, 2, 4], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]

# train Padding:  [[ 0  0  0  5  3  2  7]
# 				  [ 0  0  0  5  3  2  4]
# 				  [ 0  0  0  6  3  2  4]
# 				  [ 8  6  9  2  4 10 11]]

# test Sequence:  [[5, 1, 3, 2, 4], [2, 4, 1, 3, 2, 1]]

# test Padding:  [[0 5 1 3 2 4]
#  				[2 4 1 3 2 1]]
