
# coding: utf-8

# In[36]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import string
import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals import joblib

import py_crepe

from keras.utils.np_utils import to_categorical
from keras.models import load_model

# In[31]:


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


# In[32]:


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
    if remove_num:
        tweet = removeNum(tweet)
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]
    if remove_hashtags:
        tokens = removeHashtags(tokens)
    if remove_swords:
        tokens = removeStopwords(tokens)
    if stem_tweet:
        tokens = stemTweet(tokens)
    text = " ".join(tokens)
    return text


# In[33]:


def load_ag_data():
    train = pd.read_csv('public_development_en/train_en.tsv', delimiter='\t', encoding='utf-8')
    train = train.dropna()
    # train = train.loc[train['HS'] == 1]

    x_train = train['text'].map(lambda x: processTweet(x, remove_swords = False, remove_url = True, 
                                remove_hashtags = False, remove_num = True, stem_tweet = False))
    x_train = np.array(x_train)

    y_train = train['HS']
    y_train = to_categorical(y_train)

    test = pd.read_csv('public_development_en/dev_en.tsv', delimiter='\t', encoding='utf-8')
    # test = test.loc[test['HS'] == 1]
    x_test = test['text'].map(lambda x: processTweet(x, remove_swords = False, remove_url = True, 
                                remove_hashtags = False, remove_num = True, stem_tweet = False))
    x_test = np.array(x_test)

    y_test = test['HS']
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = set(list(string.ascii_lowercase) + list(string.digits) +
                   list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet


# In[34]:


(x_train, y_train), (x_test, y_test) = load_ag_data()


# In[38]:


np.random.seed(123)  # for reproducibility

# set parameters:

subset = None

# Whether to save model parameters
save = False
model_name_path = 'params/crepe_model.json'
model_weights_path = 'params/crepe_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 1014

# Model params
# Filters for conv layers
nb_filter = 256
# Number of units in the dense layer
dense_outputs = 1024
# Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
# Number of units in the final output layer. Number of classes.
cat_output = 2

# Compile/fit params
batch_size = 80
nb_epoch = 20

vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()
model = py_crepe.create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                              nb_filter, cat_output)


# In[40]:


x_train = encode_data(x_train, maxlen, vocab)
x_test = encode_data(x_test, maxlen, vocab)


# In[41]:


model.fit(x_train, y_train,
          validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, shuffle=True)


y_predict = model.predict(x_test, batch_size=None, steps=None)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

print(precision_score(y_test, y_predict, average=None))
print(recall_score(y_test, y_predict, average=None))
print(f1_score(y_test, y_predict, average=None))
print(roc_auc_score(y_test, y_predict, average=None))
# acc = 0
# for i in range(y_predict.shape[0]):
#     if y_predict[i] == y_test[i]:
#         acc += 1

# print(acc/y_predict.shape[0])
