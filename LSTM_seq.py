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
import lstm_utils as lu 

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

data_path = os.path.join(project_root, "data") + '/'

wastate_db = data_path + 'ccer_data.db'

# WIP: Function to connect to db and load data
engine = create_engine(f"sqlite:///{wastate_db}", echo=True)
sqlite_conn = engine.connect()

model_df = pd.read_sql_table(
    'sequence_processed',
    con=sqlite_conn
)

sqlite_conn.close()

# use the subject class as the label
# For encoder layer
# 1. Split the data into Test, Train
# 2. Run encoder on label and text pairs 

crs_seq = model_df['course_seq']

label = model_df['cadr_sum']

# Testing for encoder 
# max sequence of course title
num_words_row = [len(words.split()) for words in crs_seq]
max_seq_len = max(num_words_row)

# Tokenize
word_index, sequences = lu.tokenize_seq(crs_seq)
print('Found %s unique tokens.' % len(word_index))
vocab_size = 350

# Padding
seq_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
seq_pad.shape

# Outcome 
y_label = to_categorical(np.asarray(label))

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(seq_pad, label,
    test_size=0.2, random_state = 42)

# Build model

model = lu.model_build(vocab_size=vocab_size)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_val, y_val)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
lu.plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
lu.plot_graphs(history, 'loss')
plt.ylim(0,None)

# WIP PREDICTIONS
# predictions = model.predict(np.array([sample_courses])) 

# Another layer
model_2 = lu.model_build(vocab_size=vocab_size, hidden_layers=2)

model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model_2.fit(x_train, y_train, epochs=30,
                    batch_size=32,
                    validation_data=(x_val, y_val))

test_loss, test_acc = model_2.evaluate(x_val, y_val)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
lu.plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
lu.plot_graphs(history, 'loss')
plt.ylim(0,None)
