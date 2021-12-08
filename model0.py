#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:45:08 2021

@author: seangao
"""

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam





# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# simple lstm+glove
# public score: 0.96921




# load data
data_train = pd.read_csv('train.csv')
data_train_text = data_train['comment_text'].fillna('DUMMY_VALUE').values

list_colname = data_train.columns.values.tolist()

list_labels = list_colname[2:]
data_targets = data_train[list_labels].values





# model configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5





# load in pre-trained word vectors
wordvectors = {}
with open('glove.6B.50d.txt') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    wordvectors[word] = vec





# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(data_train_text)
data_sequences = tokenizer.texts_to_sequences(data_train_text)

word2idx = tokenizer.word_index

data = pad_sequences(data_sequences, maxlen=MAX_SEQUENCE_LENGTH)





# prepare embedding matrix
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = wordvectors.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector





# load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)





# build model
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(LSTM(15, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(list_labels), activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy'],
)




# train model
r = model.fit(
  data,
  data_targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)





# predict
data_test = pd.read_csv('test.csv')
data_test_text = data_test['comment_text'].fillna('DUMMY_VALUE').values
data_test_sequences = tokenizer.texts_to_sequences(data_test_text)
data_test = pad_sequences(data_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

p = model.predict(data_test, verbose=1)





# generate submission
sub_sample = pd.read_csv('sample_submission.csv')
sub = pd.DataFrame(p, columns=list_labels)
sub['id'] = sub_sample['id']
sub = sub[['id'] + list_labels]

sub.to_csv('submission.csv', index=False)