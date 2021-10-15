# Read all the libraries which we're going to use in our bot & Load Data
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Numpy is a matrices libraries
import numpy as np
from sklearn.utils import shuffle

## Import Keras which is Deep Learning library.
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD

## Import tensorflow for using Keras, instead.
import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(f"Num GPUs Available: {len(physical_devices)}")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Import Natural Language Processing toolkit
import nltk
# nltk.download()    # for missing language interpretable
from nltk.stem import WordNetLemmatizer

## Import jason files and pickles to read our training datasets
import json
import pickle

# create an instance from nltk
lemmatizer = WordNetLemmatizer()

# opens & read the intents JSON file dataset for data training
intents_file = open(os.path.abspath('intents.json')).read()
intents = json.loads(intents_file)

