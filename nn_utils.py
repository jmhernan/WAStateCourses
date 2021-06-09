# LSTM UTILS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# WIP: Incorporate stop rules using callback limit...maybe once accuracy 
# doesn't change by
# by some factor. 
class callbackLimit():
    pass

def lstm_model_build(vocab_size, embedding_dim=64, dropout=.50, nodes = 100, 
                     embedding_matrix=None):
    """ Builds and returns LSTM model
    Parameters:
    ----------
    vocab_size: int
        the size of your corpus vocabulary after cleanup, usually we use 
        (size_of_vocab + 1)
    
    embedding_dim: int
        this is the size of your embedding space per word, if you have a
        pre-trained embedding matrix then you have to make sure the embedding
        dimensions match. Defaults to 64
    
    dropout: float
    
    nodes: int

    embedding_matrix: numpy narray
        A word by word embedding matrix with weights for the whole vocabulary
        with dimension (vocabulary size x embedding vector). Defaults to None
    
    Returns:
    -------
    Tensorflow model object
    """
    if embedding_matrix is not None:
        print('Running pre-trained model build')
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim = vocab_size, 
                                  output_dim  = embedding_dim, 
                                  embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), 
                                  trainable = False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        print(model.summary())
    else:
        print('Running regular model build')
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        print(model.summary())
    return model


def train_model(model):
    pass

# Tokenize WIP not returning desired output...
def tokenize_seq(input_txt):
    tokenizer = Tokenizer(filters=' ')
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

def populate_embeddings(vocabulary_index, word_vector_dim, wv_model):
    """ Populates a word embedding matrix with trained weights
    Parameters:
    ----------
    vocabulary_index: dict
        A dictionary of every word in your corpus and its vector representation (int)
    
    word_vector_dim: int
        the size of the embedding space per word

    wv_model: KeyedVectors 
        Word2Vec model object of trained word embeddings that contain the 
        mapping between words and embeddings. Usually loaded in binary format
    
    Returns:
    -------
    a numpy.ndarray with a mapping of the words in your corpus and their
    embeddings
    """
    embedding_matrix = np.zeros((len(vocabulary_index) + 1, word_vector_dim))
    for word, i in vocabulary_index.items():
        if i >= len(vocabulary_index) + 1:
            continue
        try:
            embedding_vector = wv_model.wv.__getitem__(word)
            embedding_matrix[i] = embedding_vector
        except KeyError: # Catch non-vocabulary KeyErrors and populate with zeros...
            embedding_matrix[i]=np.zeros(word_vector_dim)
    return embedding_matrix