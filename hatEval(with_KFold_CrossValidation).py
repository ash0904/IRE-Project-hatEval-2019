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

from sklearn.model_selection import KFold

# functions for cleaning
def removeStopwords(tokens):
    stops = set(stopwords.words("english"))
    stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    toks = [tok for tok in tokens if not tok in stops and len(tok) >= 3]
    return toks

def removeURL(text):
    newText = re.sub('http\\S+', '', text, flags=re.MULTILINE)
    return newText

def removeNum(text):
    newText = re.sub('\\d+', '', text)
    return newText

def removeHashtags(tokens):
    toks = [ tok for tok in tokens if tok[0] != '#']
#     if segment == True:
#         segTool = Analyzer('en')
#         for i, tag in enumerate(self.hashtags):
#             text = tag.lstrip('#')
#             segmented = segTool.segment(text)

    return toks

def stemTweet(tokens):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words


def processTweet(tweet, remove_swords = True, remove_url = True, remove_hashtags = True, remove_num = True, stem_tweet = True):
#     text = tweet.translate(string.punctuation)   -> to figure out what it does ?
    """
        Tokenize the tweet text using TweetTokenizer.
        set strip_handles = True to Twitter username handles.
        set reduce_len = True to replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    if remove_url:
        tweet = removeURL(tweet)
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]
    if remove_hashtags:
        tokens = removeHashtags(tokens)
    if remove_swords:
        tokens = removeStopwords(tokens)
    if stem_tweet:
        tokens = stemTweet(tokens)
    text = " ".join(tokens)
    return text


train_data = pd.read_csv('./public_development_en/train_en.tsv',delimiter='\t',encoding='utf-8')
# print(list(train_data.columns.values)) #file header

# tweets = train_data['text']
maxlen = 140
train_data['text'] = train_data['text'].map(lambda x: processTweet(x))
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
seed = 10
fold = 1

kfold = KFold(n_splits=10)
kfold.get_n_splits(X_train)
cvscores = []

for train, test in kfold.split(X_train):
    print("------------------------------------------------------------------------------")
    print("FOLD ", fold)

    tweet = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(V, 128, input_length=maxlen, W_regularizer=l2(l=l2_coef))(tweet)
    x = Bidirectional(layer=GRU(128, return_sequences=False, 
                                W_regularizer=l2(l=l2_coef),
                                b_regularizer=l2(l=l2_coef),
                                U_regularizer=l2(l=l2_coef)),
                      merge_mode='sum')(x)
    x = Dense(len(set(labels)), W_regularizer=l2(l=l2_coef), activation="softmax")(x)

    tweet2vec = Model(inputs=tweet, outputs=x)

    tweet2vec.compile(loss='categorical_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])


    tweet2vec.fit(X_train[train], Y_train[train], epochs=10, batch_size=32, validation_split=0.1)


    scores = tweet2vec.evaluate(X_train[test], Y_train[test], verbose=0)
    print("%s: %.2f%%" % (tweet2vec.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    fold += 1
print("------------------------------------------------------------------------------")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


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