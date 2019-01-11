#THIS IS NOT MY CODE, I JUST RAN IT FOR TEST
import numpy as np
import pylab as pl
from IPython.display import SVG
import os.path
import os

from gensim.models import word2vec, KeyedVectors
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils.vis_utils import model_to_dot

#model hyperparams
embeding_dim = 300
filter_sizes = (3,4,5)
num_filters = 100
dropout_prob = (0.0, 0.5)

#training
batch_size = 64
num_epochs = 10

#preprocessing
sequence_length = 20
max_words  = 5000

#word2vec parameters
min_word_count = 1
context = 10

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None, oov_char=None, index_from=None)
x_train = x_train[:1500]
y_train = y_train[:1500]
x_test = x_test[:10000]
y_test = y_test[:10000]

x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

vocabulary = imdb.get_word_index()
vocabulary_inv = dict((v,k) for k,v in vocabulary.items())
vocabulary_inv[0] = "<PAD/>"
print("x_train shape:", x_train )
print("x_test shape:", x_test)
print("Vocabulary size: {:d}".format(len(vocabulary_inv)) )

#Input Layer
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

#Embedding Layer
embedding_layer = Embedding(len(vocabulary_inv), embeding_dim, input_length=sequence_length, name="embedding")
z = embedding_layer(model_input)

#Conv blockss
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks if len(conv_blocks) > 1 else conv_blocks[0])
z = Dropout(dropout_prob[1])(z)
model_output = Dense(1, activation="sigmoid")(z)
model_rand = Model(model_input, model_output)
model_rand.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_rand.summary(85)
SVG(model_to_dot(model_rand, show_shapes=True).create(prog='dot', format='svg'))
num_epochs = 10
history_rand = model_rand.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)
pl.plot(history_rand.history['loss'], label='loss')
pl.plot(history_rand.history['val_loss'], label='val_loss')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Loss')
pl.plot(history_rand.history['acc'], label='acc')
pl.plot(history_rand.history['val_acc'], label='val_acc')
pl.legend()
pl.xlabel('Epoch')
pl.ylabel('Accuracy')

num_epochs = 10
history_rand = model_rand.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

#for static and non static CCN
'''embedding_model = KeyedVectors.load_word2vec_format()
'''