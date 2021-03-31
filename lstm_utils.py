# LSTM UTILS
import matplotlib.pyplot as plt

class callbackLimit():
    pass

def model_buid():
    pass

def train_model():
    pass


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

