# LSTM UTILS
import matplotlib.pyplot as plt

class callbackLimit():
    pass

def model_buid():
    pass

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

