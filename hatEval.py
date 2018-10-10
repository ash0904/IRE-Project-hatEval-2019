import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, GRU, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.utils import np_utils



def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text

train_data = pd.read_csv('./public_development_en/train_en.tsv',delimiter='\t',encoding='utf-8')
# print(list(train_data.columns.values)) #file header

# tweets = train_data['text']
maxlen = 140
train_data['text'] = train_data['text'].map(lambda x: clean_text(x))
vocabulary_size = 30000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(train_data['text'])
X_train = tokenizer.texts_to_sequences(train_data['text'])
# print(sequences)
X_train = pad_sequences(X_train, maxlen=maxlen)
labels = train_data['HS']
Y_train = np_utils.to_categorical(labels, len(set(labels)))
V = len(tokenizer.word_index) + 1


l2_coef = 0.001
tweet = Input(shape=(maxlen,), dtype='int32')
x = Embedding(V, 128, input_length=maxlen, W_regularizer=l2(l=l2_coef))(tweet)
x = Bidirectional(layer=GRU(128, return_sequences=False, 
                            W_regularizer=l2(l=l2_coef),
                            b_regularizer=l2(l=l2_coef),
                            U_regularizer=l2(l=l2_coef)),
                  merge_mode='sum')(x)
x = Dense(len(set(labels)), W_regularizer=l2(l=l2_coef), activation="softmax")(x)

tweet2vec = Model(input=tweet, output=x)

tweet2vec.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])


tweet2vec.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)


'''
embeddings_index = dict()
f = open('./glove.twitter.27B/glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

print(tokenizer.word_index.items())

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 200, input_length=50, weights=[embedding_matrix], trainable=True))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(200))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model_glove.fit(data, np.array(labels), validation_split=0.3, epochs = 20)
'''