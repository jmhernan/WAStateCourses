# MODEL CODE FOR CADRS MODEL ADAPT TO SEQUENCE OUTPUT

import os
import re
import random
import sys
from pathlib import Path

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
num_words_row = [len(words.split()) for words in text]
max_seq_len = max(num_words_row)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))
vocab_size = 589

sequences = tokenizer.texts_to_sequences(text)

# Padding
text_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
text_pad.shape

# Handling the label
encoder = LabelEncoder()
label_arr = encoder.fit_transform(label)

num_classes = np.max(label_arr) + 1 #index at 0
label_pad = utils.to_categorical(label_arr, num_classes)

print('Shape of data tensor:', text_pad.shape)
print('Shape of label tensor:', label_pad.shape)

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(text_pad, label_pad,
    test_size=0.2, random_state = 42)

# Build a model fun
embedding_dim = 100
dropout = .25


model = tf.keras.Sequential([
    # Embedding layer
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # Dense layer
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # output layer
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
