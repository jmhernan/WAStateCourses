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
    'tuk_sequence_processed',
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

model_df = load_sql_table(table_name='tuk_sequence_processed', db_name=wastate_db)
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

print('Found %s unique tokens.' % len(word_index))
vocab_size = len(word_index) + 1

# Padding
seq_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
seq_pad.shape

# Outcome 
y_label = np.asarray(label)

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(seq_pad, y_label,
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
nn.plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
nn.plot_graphs(history, 'loss')
plt.ylim(0,None)

# WIP PREDICTIONS
# predictions = model.predict(np.array([sample_courses])) 

# Word embeddings 
embeddings_test = '/Users/josehernandez/Documents/eScience/projects/WAStateCourses/seqcrs/course_baseline_model_test.bin'
embeddings_aws = '/home/ubuntu/source/WAStateCourses/seqcrs/course_baseline_model.bin'

model = Word2Vec.load(datapath(embeddings_test))
model.wv.most_similar('fine_arts', topn=10) #WIP double underscore
model.wv.__getitem__('algebra_1')
# create embedding matrix
word_vector_dim=100
embedding_matrix = np.zeros((len(word_index) + 1, word_vector_dim))


def populate_embeddings(vocabulary_index, word_vector_dim, wv_model):
    embedding_matrix = np.zeros((len(vocabulary_index) + 1, word_vector_dim))
    for word, i in word_index.items():
        if i >= len(vocabulary_index) + 1:
            continue
        try:
            embedding_vector = wv_model.wv.__getitem__(word)
            embedding_matrix[i] = embedding_vector
        except KeyError: # Catch non-vocabulary KeyErrors and populate with zeros...
            embedding_matrix[i]=np.zeros(word_vector_dim)
    return embedding_matrix

emb_matrix = populate_embeddings(vocabulary_index=word_index,word_vector_dim=100, wv_model = model)

# CHECK ZERO ENTRIES
nonzero_elements = np.count_nonzero(np.count_nonzero(emb_matrix, axis=1))
nonzero_elements / len(word_index)

# WIP: add to model 
vocab_size = len(word_index) + 1
embedding_dim=100 
dropout=.50 
nodes = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = vocab_size, output_dim  = embedding_dim, embeddings_initializer = tf.keras.initializers.Constant(emb_matrix), trainable = False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val))
