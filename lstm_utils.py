# LSTM UTILS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# WIP: Incorporate stop rules using callback limit...maybe once accuracy doesn't change by
# by some factor. 
class callbackLimit():
    pass

def model_build(vocab_size, embedding_dim=64, dropout=.50, hidden_layers=1, nodes = 100):
    if hidden_layers > 1:
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    return model


def train_model():
    pass

# Tokenize
def tokenize_seq(input_txt):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input_txt)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(input_txt) # word and their token # ordered by most frequent
    return word_index, sequences
    
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

