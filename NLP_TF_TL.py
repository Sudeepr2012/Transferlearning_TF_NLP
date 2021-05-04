import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
from  IPython import display

import pathlib
import shutil
import tempfile

!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

print("Version: ", tf.__version__)
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

data=pd.read_csv("https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip", compression='zip', low_memory=False)
data.shape
data["target"].plot(kind="hist", title="Target Distribution")

from sklearn.model_selection import train_test_split
train_data, remaining= train_test_split(data, random_state=42, train_size=0.2, stratify=data.target.values)
valid_data, _=train_test_split(remaining, random_state=42, train_size=0.001, stratify=remaining.target.values)
train_data.shape, valid_data.shape

train_data.question_text.head(15).values


module_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" #@param ["https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", "https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"] {allow-input: true}

#model function:

def train_and_evaluate_model(module_url, embed_size, name, trainable=False):
  hub_layer=hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype=tf.string, trainable=trainable)
  model=tf.keras.Sequential([
                             hub_layer,
                             tf.keras.layers.Dense(256,activation="relu"),
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")
  ])
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                metrics=["accuracy"])
  history=model.fit(train_data['question_text'], train_data['target'], epochs=100, batch_size=32, validation_data=(valid_data['question_text'], valid_data['target']),
                    callbacks=[tfdocs.modeling.EpochDots(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode='min'),
                               tf.keras.callbacks.TensorBoard(logdir/name)], verbose=0)
  return history  

histories={}

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5" #@param ["https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", "https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"] {allow-input: true}
histories['universal-sentence-encoder-large']= train_and_evaluate_model(module_url, embed_size=512, name='universal-sentence-encoder-large')

#plot accuracy curves
plt.rcParams['figure.figsize'] = (12, 8)
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Accuracy Curves for Models")
plt.show()

#plot loss curves

plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Loss Curves for Models")
plt.show()

#using tensorboard
%load_ext tensorboard
%tensorboard --logdir {logdir}
