import tensorflow as tf
import tensorflow_datasets as tfds
imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

print(tokenizer.subwords)

sample_string = "Tensorflow, from basics to mastery"

tokenized_string = tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('the original string: {}'.format(original_string))

for i in tokenized_string:
  print("{} ----> {}".format(i, tokenizer.decode([i])))

embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# only if error occur at fitting te model
BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes))

test_batches = (
    test_data
    .padded_batch(32,train_data.output_shapes))

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

#########################################################

num_epochs = 10

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
history = model.fit(train_batches, epochs=num_epochs, validation_data=test_batches, validation_steps=30)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

