import  pandas  as  pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import scipy.sparse as sp
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import operator
from keras.callbacks import History
from keras.datasets import imdb
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
from keras.models import load_model

from keras.models import model_from_json
MAX_SEQUENCE_LENGTH = 1000
tokenizer = joblib.load('tokenizer.pickle')
df_reviews = pd.read_csv('/Users/Devika/Desktop/project/newtest5.csv')#, encoding='utf-8')
seqs = tokenizer.texts_to_sequences(df_reviews.Text.values)
X = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
# Model reconstruction from JSON file
with open('/Users/Devika/Desktop/project/model_num.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('/Users/Devika/Desktop/project/model_num.h5')
preds = model.predict_classes(X)
print(preds)