# LSTM UTILS
import matplotlib.pyplot as plt

class callbackLimit():
    pass

def model_build(vocab_size, embedding_dim=64, dropout=.25, hidden_layers=1):
    if hidden_layers > 1: # WIP: Need to add different hidden layer conditions
        # DO CV TO COMPARE AND FINETUNE IN GRID SEARCH METHOD
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1)])
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
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
    word_index = tokenizer.word_index # word and their token # ordered by most frequent
    return word_index
    

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

