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
    ### Preprocessing
	The first big challenge was to clean and process the tweets to remove the noise 
and other unnecessary words, which would be quite useful for training any model and would be helpful in producing better results. Hopefully, we got to know about TweetTokenizer which is part of NLTK library, we used it to remove Twitter username handles and replace repeated character sequences of length 3 or greater, with sequences of length 3 which generally present as noise in tweets. Then there we thought of experimenting our models by modifying the tweets in a few different ways and hence we wrote functions to try out following:
	+ Remove_URL: process tweets with and without removal of URL
	+ Remove_Hashtags: process tweets with and without removal of Hashtag
	+ Remove_num:  process tweets with and without removal of Numbers
	+ Remove_Swords: process tweets with and without removal of Stopwords
	+ Stem_tweet: process tweets with and without stemming of words

	We found that the accuracy of various models change when we use the above functions to modify the tweets, in general on a positive side. Preprocessing procedure depends upon the model being used, we used all of the above functions for training our model.

