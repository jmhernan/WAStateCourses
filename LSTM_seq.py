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

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec
from gensim.test.utils import datapath

from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
this_file_path = '/home/ubuntu/source/WAStateCourses/LSTM_seq.py'
project_root = os.path.split(this_file_path)[0]

sys.path.insert(1, project_root)

import nn_utils as nn

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

def load_sql_table(table_name, db_name):
    engine = create_engine(f"sqlite:///{db_name}", echo=False) # Find a way to use this in messages
    sqlite_conn = engine.connect()
    df = pd.read_sql_table(
        table_name,
        con=sqlite_conn
    )
    sqlite_conn.close()
    return df

model_df = load_sql_table(table_name='sequence_processed', db_name=wastate_db)
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
word_index, sequences = nn.tokenize_seq(crs_seq)
# TRY
tokenizer = Tokenizer(filters=' ')
tokenizer.fit_on_texts(crs_seq)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(crs_seq) # word and their token # ordered by most frequent

print('Found %s unique tokens.' % len(word_index))
vocab_size = 450

# Padding
seq_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
seq_pad.shape

# Outcome 
y_label = to_categorical(np.asarray(label))

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(seq_pad, label,
    test_size=0.2, random_state = 42)

# Build model

model = nn.lstm_model_build(vocab_size=vocab_size)

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

# Word embeddings 
model = Word2Vec.load(datapath('/home/ubuntu/source/WAStateCourses/seqcrs/course_baseline_model.bin'))
model.wv.most_similar('fine_arts', topn=10) #WIP double underscore
model.__getitem__('algebra_1')
# create embedding matrix
word_vector_dim=100
embedding_matrix = np.zeros((len(word_index) + 1, word_vector_dim))

# WIP
for word, i in word_index.items():
    if i >= len(word_index) + 1:
        continue
    try:
        embedding_vector = model.__getitem__[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),word_vector_dim)
# WIP Fix tokenizer
crs_seq
tokenizer = Tokenizer(filters=' ')
tokenizer.fit_on_texts(crs_seq)
word_index = tokenizer.word_index