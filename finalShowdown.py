import  pandas  as  pd
import h5py
#import os
import csv
#os.chdir("\Users\Devika\Desktop\project")
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
import keras.backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
np.random.seed(0)
VALIDATION_SPLIT = 0.2
GLOVE_DIR = '/Users/Devika/Desktop/project/glove.6B.100d.txt'
EMBEDDING_DIM = int(''.join([s for s in GLOVE_DIR.split('/')[-1].split('.')[-2] if s.isdigit()])) # 100

def load_glove_into_dict(glove_path):
    """
    :param glove_path: strpath
    loads glove file into a handy python-dict representation, where a word is a key with a corresponding N-dim vector
    http://nlp.stanford.edu/data/glove.6B.zip (pretrained-embeddings)
    """
    embeddings_ix = {}
    with open(glove_path,encoding="utf8") as glove_file:
        for line in glove_file:
            val = line.split()
            word = val[0]
            vec = np.asarray(val[1:], dtype='float32')
            embeddings_ix[word] = vec
    return embeddings_ix
	
	
def minority_balance_dataframe_by_multiple_categorical_variables(df, categorical_columns=None, downsample_by=0.1):
    """
    :param df: pandas.DataFrame
    :param categorical_columns: iterable of categorical columns names contained in {df}
    :return: balanced pandas.DataFrame
    """
    if categorical_columns is None or not all([c in df.columns for c in categorical_columns]):
        raise ValueError('Please provide one or more columns containing categorical variables')

    minority_class_combination_count = df.groupby(categorical_columns).apply(lambda x: x.shape[0]).min()
    
    minority_class_combination_count = int(minority_class_combination_count * downsample_by)
    
    df = df.groupby(categorical_columns).apply(
        lambda x: x.sample(minority_class_combination_count)
    ).drop(categorical_columns, axis=1).reset_index().set_index('level_1')

    df.sort_index(inplace=True)

    return df


df_reviews = pd.read_csv('/Users/Devika/Desktop/project/Reviews.csv', encoding='utf-8')
df_reviews['len'] = df_reviews.Text.str.len()
df_reviews = df_reviews[df_reviews['len'].between(10, 4000)]
#     df_reviews = df_reviews[df_reviews.language == 'en']
# balancing dataset
df_rev_balanced = minority_balance_dataframe_by_multiple_categorical_variables(
    df_reviews, 
    categorical_columns=['Score'], 
    downsample_by=0.1
)
    
df_rev_balanced.to_csv('balanced_reviews.csv', encoding='utf-8')
    
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df_rev_balanced.Text.tolist())
joblib.dump(tokenizer, 'tokenizer.pickle')

WORD_INDEX_SORTED = sorted(tokenizer.word_index.items(), key=operator.itemgetter(1))
    
seqs = tokenizer.texts_to_sequences(df_rev_balanced.Text.values)
'''
X = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
Y = df_rev_balanced.Score.values.astype(int)
Y_cat = to_categorical(Y)
assert X.shape[0] == Y.shape[0]  
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=9)

with pd.HDFStore('x_y_test_train.h5') as h:
	h['X_train'] = pd.DataFrame(X_train)
	h['X_test'] = pd.DataFrame(X_test)
	h['y_train'] = pd.DataFrame(y_train)
	h['y_test'] = pd.DataFrame(y_test)	
print(X_train)
print(y_train)
'''


filename1 = '/Users/Devika/Desktop/project/x_y_test_train.h5'
data = h5py.File(filename1, 'r')
X_train = data['X_train']['block0_values'].value
X_test=data['X_test']['block0_values'].value
y_train=data['y_train']['block0_values'].value
y_test=data['y_test']['block0_values'].value




embeddings_index = load_glove_into_dict(GLOVE_DIR)
nb_words = min(MAX_NB_WORDS, len(WORD_INDEX_SORTED))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector   
		
filepath="/Users/Devika/Desktop/project/imp-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('/Users/Devika/Desktop/project/training_history.csv')
history = History()
callbacks_list = [checkpoint, history, csv_logger]		
model = Sequential()

model.add(Embedding(input_dim=nb_words, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    weights=[embedding_matrix], 
                    trainable=False)
)

model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy',root_mean_squared_error])
print(model.summary())
model.fit(X_train, 
          y_train, 
          nb_epoch=10, 
          batch_size=128, 
          validation_data=(X_test, y_test),
		  callbacks=callbacks_list
        
)
preds = model.predict_classes(X_test)
print(preds)
print(X_test)
model_json = model.to_json()


with open("/Users/Devika/Desktop/project/model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("/Users/Devika/Desktop/project/model_num.h5")