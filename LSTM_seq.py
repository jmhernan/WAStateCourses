# MODEL CODE FOR CADRS MODEL ADAPT TO SEQUENCE OUTPUT
import os
import re
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import csv
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

data_path = os.path.join(project_root, "data") + '/'

wastate_db = data_path + 'ccer_data.db'

# connect to db and load data
engine = create_engine(f"sqlite:///{wastate_db}", echo=True)
sqlite_conn = engine.connect()

model_df = pd.read_sql_table(
    'sequence_processed',
    con=sqlite_conn
)

sqlite_conn.close()

# use the subject class as the label
crs_seq = model_df['course_seq']


label = model_df['cadr_sum'].tolist()

# max sequence of course title
num_words_row = [len(words.split()) for words in crs_seq]
max_seq_len = max(num_words_row)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(crs_seq)
word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))
vocab_size = 350

sequences = tokenizer.texts_to_sequences(crs_seq)

# Padding
seq_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
seq_pad.shape

y_label = to_categorical(np.asarray(label))

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(text_pad, label,
    test_size=0.2, random_state = 42)

# Build a model
embedding_dim = 64
dropout = .25


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_val, y_val), 
                    validation_steps=30)

test_loss, test_acc = model.evaluate(x_val, y_val)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plot_graphs(history, 'acc')
plt.ylim(None,1)
plt.subplot(1,2,2)
plot_graphs(history, 'loss')
plt.ylim(0,None)

