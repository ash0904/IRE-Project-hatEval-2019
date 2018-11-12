## Hate speech Detection against immigrants and women in tweets
                                          SemEval19: hatEval

## Introduction
Hate Speech is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics. Given the huge amount of user-generated contents on the Web, and in particular on social media, the problem of detecting, and therefore possibly limit the Hate Speech diffusion, is becoming fundamental, for instance for fighting against misogyny and xenophobia.

The proposed task consists of Hate Speech detection in Twitter but featured by two specific different targets, immigrants and women.

The task was articulated around two related subtasks for each of the involved languages: a basic task about Hate Speech, and another one where fine-grained features of hateful contents will be investigated in order to understand how existing approaches may deal with the identification of especially dangerous forms of hate, i.e. those where the incitement is against an individual rather than against a group of people, and where an aggressive behaviour of the author can be identified as a prominent feature of the expression of hate.

## Tasks

Our project was divided into two major tasks :-
1. **Hate Speech Detection against Immigrants and Women:** a two-class (or binary) classification where systems have to predict whether a tweet with a given target (women or immigrants) is hateful or not hateful.
2. **Aggressive behaviour and Target Classification:** where systems are asked first to classify hateful tweets (e.g., tweets where Hate Speech against women or immigrants has been identified) as aggressive or not aggressive, and second to identify the target harassed as individual or generic (i.e. single human or group).

## Dataset

We registered for participating in the [Semeval contest 2019](https://competitions.codalab.org/competitions/19935 "hatEval"), and were provided with the data to train our model.
[Link to the dataset](https://github.com/ash0904/IRE-Project-hatEval-2019/tree/master/public_development_en)

**Details**
+ 8100 tweets for training.
+ 900 tweets for testing.
+ Nearly 3700 tweets contained hate.

## Technologies Used :

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/) - Pandas package provides fast, flexible and expressive data structures designed to make working with “relational” or “labelled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical,real-world data analysis. Hence it was very useful tool for parsing the dataset provided.
* [Scikit-Learn](http://scikit-learn.org/stable/index.html) - Scikit learn is an efficient tool with the implementation of a class of data analysis and machine learning algorithms. Hence it made our life easy by providing functions for calculating various accuracy parameters like precision, recall, F1_score, roc-auc score for validating our model.
* [Keras](https://keras.io/) Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. We coded our neural networks in Keras with the backend as TensorFlow.

## Approach
1. **Preprocessing :-** The first big challenge was to clean and process the tweets to remove the noise 
and other unnecessary words, which would be quite useful for training any model and would be helpful in producing better results. Hopefully, we got to know about TweetTokenizer which is part of NLTK library, we used it to remove Twitter username handles and replace repeated character sequences of length 3 or greater, with sequences of length 3 which generally present as noise in tweets. Then there we thought of experimenting our models by modifying the tweets in a few different ways and hence we wrote functions to try out following:
	+ Remove_URL: process tweets with and without removal of URL
	+ Remove_Hashtags: process tweets with and without removal of Hashtag
	+ Remove_num:  process tweets with and without removal of Numbers
	+ Remove_Swords: process tweets with and without removal of Stopwords
	+ Stem_tweet: process tweets with and without stemming of words
We found that the accuracy of various models change when we use the above functions to modify the tweets, in general on a positive side. Preprocessing procedure depends upon the model being used, we used all of the above functions for training our model.

2. **Word Embedding Model :-** A word embedding is a class of approaches for representing words and documents using a dense vector representation. Words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space. The position of a word within the vector space is learned from the text and is based on the words that surround the word when it is used. The position of a word in the learned vector space is referred to as its embedding.
    + First, data is prepared using the Tokenizer API also provided with Keras where the tweets are converted into a list of sequences (one per text input). These sequences are padded to make all of them of the same size. The list of sequences is transformed into a 2D Numpy array of shape (number_of_tweets, MAXLEN). MAXLEN if not provided, is the length of the longest sequence otherwise. Sequences that are shorter than MAXLEN are padded with value at the end and sequences longer than MAXLEN are truncated so that they fit the desired length. The vector of expected outcome is converted into a binary matrix representation of it.
    + Embedding layer is used as the first layer in the model. It requires that the input data be integer encoded so that each word is represented by a unique integer. The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset. The output of the Embedding layer is a 2D vector with one embedding for each word in the input sequence of words.
    + Next, a Bidirectional layer is added. The input sequence is fed in normal time order for one network, and in reverse time order for another. The outputs of the two networks are usually concatenated at each time step. This structure allows the networks to have both backward and forward information about the sequence at every time step.
    + Lastly, a Dense layer is introduced where we have used softmax as an activation function. A dense layer is just a regular layer of neurons in a neural network. Each neuron receives input from all the neurons in the previous layer, thus densely connected.
3. **Character level CNN :-** The next model we tried was character based CNN since texts in tweet corpus is a bit different than any other regular text corpus. In tweets, word shorthands e.g. ‘FYKI’ instead of ‘for your kind information’, repeating characters e.g. ‘yaaayyyyyy’ to express feelings, emoticons using symbols etc are used. These are not general English words and these usages may differ from person to person. As for example, ‘yaaaayyyyy’ and ‘yaayyy’ both are the same word. So character level model would work better than word level model. 
In this approach, we came up with the idea of embedding the characters of the words and then pass it through the CNN network.
    + The model accepts a sequence of encoded characters as input. The encoding is done by prescribing an alphabet of size m for the input language, and then quantize each character using 1-of-m encoding (or “one-hot” encoding). Then, the sequence of characters is transformed to a sequence of such m sized vectors with fixed length l0. Any character exceeding length l0 is ignored, and any characters that are not in the alphabet including blank characters are quantized as all-zero vectors. The character quantization order is backward so that the latest reading on characters is always placed near the begin of the output, making it easy for fully connected layers to associate weights with the latest reading.
    ![CNN for language](https://raw.githubusercontent.com/ash0904/Algorithms/master/images/cnn1.png)
    + The alphabet used in all of our models consists of 70 characters, including 26 english letters, 10 digits, 33 other characters and the new line character. The non-space characters are: abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|@#$%ˆ&˜‘+-=<>()[]{}
    + The embeddings are then passed to the CNN network whose architecture is described below. We also insert 2 dropout modules in between the 3 fully-connected layers to regularize. They have dropout probability of 0.5.     
    ![Model Archietecture](https://raw.githubusercontent.com/ash0904/Algorithms/master/images/model1.png)
