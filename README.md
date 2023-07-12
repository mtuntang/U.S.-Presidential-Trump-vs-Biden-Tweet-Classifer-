# U.S. Presidential (Trump-vs-Biden) Tweet Classifer

# Description
This is a model consisting of a number binary classifiers which,
given a tweet vector, predict whether the tweet was authored by Donald Trump or Joe Biden.

# Data
The dataset is a list of tweets from both Joe Biden and Donald J. Trump extracted by a word2vec algorithm combined via averaging
to form a fixed-length feature vector per tweet. 
- wordvec train.csv is a dataset of word2vec features. Each data entry has 200 numerical features, and
a label (1 for Donald Trump, and 0 for Joe Biden). This is the training data.
- tweet train.csv is a dataset of tweets prior to feature extraction (you don't need to use this, it is
provided just for own amusement). These correspond to features in the training data.
- wordvec test.csv is dataset of word2vec features. This is the test set.
- tweet test.csv consists of tweets corresponding to test set features.

# Classifiers Implemented
1. A random forest classifier.
     - Which uses the Gini index as a splitting score.
     - Where the K-means algorithm is used to quantize each feature, determining the thresholds to
search over for each feature.
2. Naive Bayes for continuous features.
3. K nearest neighbours with cosine similarity as the distance metric.
4. A stacking ensemble consisting of your classifiers from 1-3, where a decision tree is used as a
meta-classifier.
