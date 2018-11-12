## Movies recommendation System

Websites such as IMDB, Amazon, Wikipedia have a number of reviews about various movies. Given general details of a movie such as genre, ratings, plot summary, director, stars, era etc along with user reviews on IMDB, Amazon and critique on Wikipedia, the goal is to design a movie recommendation system, in which a user describes the kind of movie he/she is interested in watching. The user’s query can be arbitrarily long and textually rich. The recommendation system should return top k results for the query.These systems typically have two steps:

1. Retrieval step: Retrieve top-k passages given a query.
2. Comprehension step: Comprehend the answer from the top-k passages retrieved.

## Example Queries with recommendations

| Query         | Recommendations|
| ------------- |:-------------:|
| Movies that start with recruitment scenes? Think along the lines of Seven Samurai or Ocean's Eleven. Movies that start with a group slowly being put together based on each member's unique skills and abilities. | Ronin, The Sting, 13 Assassins, Armageddon, The Dirty Dozen, The Italian Job, The Magnificent Seven |
| Are there any big budget movies about the Bolshevik Revolution? I don't know much about the implementation of communism in russia as a moment in history and think it would make a fascinating topic for a movie, just like how the pianist or schindler's list were a window into the holocaust. Think about it, there was this whole new ideology with profound societal implications that only existed in theory and some guys just got together and forced it to work through a revolution and pretty much mob rule. There is so much content in this area of history! Does anyone have any recommendations for me?| S. Eisenstein, Reds, Anastasia, Doctor Zhivago, Battleship Potemkin, Nicholas & Alexandra, The Last Station |


## Motivation
Movie recommendation is a popular problem. There are a lot of implementations available which are based on characteristics of movies and/or use collaborative filtering method to recommend movies to a user based on the movies watched or liked by other similar users.

A large amount of information exists in reviews written by users. This source of information has been ignored by most of the current recommender systems while it can potentially alleviate the sparsity problem and improve the quality of recommendations.

Our problem is a novel one. We go a step further to analyse the user reviews for a movie. Moreover, the user while specifying a query, can be as specific as they want to, our results would cater to that requirement.

Currently we have various Question Answering threads such as on
https://www.reddit.com/r/MovieSuggestions &
https://movies.stackexchange.com/ etc.
These include relevant movies as answers or comments by humans. Hence this the action that we want to automate. The idea behind using reviews is that reviews depict what a user feels about a movie. When asked a query like, this is the kind of feeling I’m looking for in a movie, the suggestions will be based on the reviews.

## Datasets

### IMDB
We crawled data for 3500 movies from IMDB. Movies on IMDB are categorised into 23 genres namely Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Short, Sport, Thriller, War and Western. We crawled around top ~ 150 movies for each genre, sorted by number of votes. Number of votes was chosen as sorting order because usually more votes means more reviews. This also eliminated movies which haven’t been released yet and hence carry no reviews. Hence for each movie, we have the following fields:

1. Genre
2. Plot
3. Directors
4. Writers
5. Stars
6. Rating
7. Similar movies
8. Plot keywords

For each movie, top 50 reviews sorted by most helpful, were collected. Only about 10 of the total movies had less than 50 reviews, but still had at least ~20 reviews. Each review is written as a title, text pair.


### Amazon product data for Movies and TV

This is a publically available [Dataset](http://jmcauley.ucsd.edu/data/amazon/). From this we extract product id and reviews for each product. We map product id to Unique movie ids collected from IMDB. The reviews are a tuple of following fields:

1. Review Text
2. Summary
3. Overall Rating.


### Wikipedia Dataset

[wiki\_data.json](https://raw.githubusercontent.com/rehassachdeva/Movie-Recommendation---IRE-Group-11/master/Data/wiki\%20data/wiki_data.json)
We crawled data from the wikipedia page of each of the 3500 movies in our database from IMDB. For each movie, we organised the data into the following fields:

1. Cinematography
2. Country
3. Critical Response
4. Directed by
5. Distributed by
6. Language
7. Produced by
8. Release date
9. Running time
10. Plot 


### Reddit Data

Reddit has a lot of threads where user describe the kind of movies they are interseted in watching, followed by a lot of comments as answers. The crawled reddit data is organised as following fields:

1. Title of post.
2. Link to the post.
3. Description of query.
4. List of comments.


### StackExchange Data
StackExchange Data was also obtained in the same way as reddit data. Only posts with positive votes were considered while crawling.

1. Title of post.
2. Link to the post.
3. Votes on comment followed by comment.


### QRel data creation
QUERY RELEVANCE SET}(qrels) for query $q$ we generate a small number $m$ of query aspects. Using a single IR system, we run each query aspect against the documents. The union of the top $k$ documents retrieved for each aspect constitutes a list of pseudo-qrels for $q$. From this point, evaluation proceeds in the usual fashion, merely substituting pseudo-qrels for human-judged qrels. The core idea is that
the set of query aspects articulates different facets of the underlying information need. Running each aspect as a query against $D$ allows us to cast a wide net, collecting what is hopefully a variety of relevant documents. We assigned relevance judgement in 3 broader categories.

1. A 0 relevance indicates that for a given movie, the recommendation is totally irrelevant for all category, details etc.
2. A relevance of 1 indicates that the movie is partially relevant to the recommendation. Example either the details or the genre is similar.
3. A relevance of 2 indicates that the movie is totally relevant to the answers of recommendations. 

## Baseline Approach
Our baseline approach is based on creating an index for the movies based on the content in their reviews. We have used Apache Solr for indexing and retrieving the Top 40 results. All the reviews from IMDB, Amazon and WIkipedia were aggregated into a document for each movie. Solr interface can then take a user query and match it to the indexed documents, returning a list of top 40 movies for the query.

## Deep learning model
We implement a deep learning based model to learn movie properties from the review text. Experimental results demonstrate that DL approach significantly outperforms all baseline approaches.

We reduce our problem to neural document ranking, considering all the content of a movie as a document. The goal then is to retrieve the most relevant documents for the given query.

At the most abstract level, document ranking is described as,

![image](https://raw.githubusercontent.com/jainayush975/IRE-MAJOR-PROJECT/master/images/image1.png)

- Neural document ranking differs in whether neural model is applied at representation stage or matching stage or both.

- The first step is to learn vector representations of every term in our vocabulary. When vector representations are dense, small and learnt from data, they are called embeddings.
Generating the embeddings is done using below methods,
  1. LSA
  2. Word2vec
  3. GloVe


- For every text in our queries + documents corpus, we aggregate its term embeddings using the below methods. Lets call the final representation from below as R.
  1. Linear combination: Average term embeddings: Take coordinate-wise mean of all the embeddings.
  2. Max method: Take coordinate-wise max of all the embeddings.
  3. Min method: Take coordinate-wise min of all the embeddings.
  4. Non-Linear combinations: Fisher Kernel framework: First compute the fisher kernel for each pair of embeddings in the text. Convert the sequence of embeddings in the text to FK values for every consecutive pair.
- Now given a query, compute it R(q). For every document in the corpus compute R(d). Rank the documents using cosine similarity.


### DSSM: Compute Similarity in Semantic Space

It consists of 5 layers.

1. Input as a sequence of terms of texts X and Y.
2. Word hashing: use sub-word unit (e.g.,letter n-gram) as raw input to handle very large vocabulary.
3. Convolutional and Max-pooling layer: identify key words/concepts in X and Y.
4. Representation: use DNN to extract abstract semantic representations.
5. Learning: maximize the similarity between X (source) and Y (target).

![image](https://raw.githubusercontent.com/jainayush975/IRE-MAJOR-PROJECT/master/images/dssm.png)

For more details about each layer refer the report.

## Evaluation Metrics
- **Precision**: Fraction of retrieved documents that are relevant.
- **Recall**: Fraction of relevant documents that are retrieved.
- **NDCG**: For a query q, let relevance level of document ranked j wrt q be rq(j).
  * rq(j)=0 means totally irrelevant
  * Response list is inspected up to rank L.
  * Discounted cumulative gain for query q is,
    * ![image](https://raw.githubusercontent.com/jainayush975/IRE-MAJOR-PROJECT/master/images/image2.png)
  * Zq is a normalization factor that ensures the perfect ordering has NDCGq = 1.
  * Overall NDCG is average of NDCGq over all q

## Challenges
- The project inherits all the challenges of text analysis such as language intricacies, and semantic analysis to derive a sufficiently rich representation to capture the relationship between the objects or concepts.
- Handling long semantic queries is computationally expensive.
- All the data was crawled
- QRel data creation

## Conclusions
Movie recommendation is a popular problem. There are a lot of implementations available which are based on characteristics of movies and/or use collaborative filtering method to recommend movies to a user based on the movies watched or liked by other similar users. Here we have used a novel approach by using data from reviews as well and also leveraged the state-of-the art research in document ranking. We have come up with a question answer based interface for movie recommendation which is more flexible and allows the user to be as specific as they want to. Currently we have search and recommendations by keywords and on a much broader level.

