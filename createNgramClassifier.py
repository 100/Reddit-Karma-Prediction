from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import os, sys, json, math, random
try:
   import cPickle as pickle
except:
   import pickle
from preprocessing import readRaw, ngramPreprocess


'''
For use with readRaw method. Extracts comment body and labels with 'popular' if
above a score of 0, otherwise 'unpopular'.
Args:
    comment (string): comment body
    texts (list of string): list to append current comment's body to
    labels (list of string): list to append current comment's classification to
'''
def seekPopular(comment, texts, labels):
    texts.append(comment['body'])
    labels.append('popular' if comment['score'] > 0 else 'unpopular')


'''
Creates and trains binary classifier with n-gram features, and pickles to disk
upon completion.
Args:
    corpus (list of string): list of comment bodies
    labels (list of string): list of comment classifications
Returns:
    classifier (sklearn classifier)
'''
def createNgramBinaryClassifier(corpus, labels):
    clf = Pipeline([
            ('vect', CountVectorizer(
                decode_error = 'replace',
                strip_accents = 'unicode',
                preprocessor = ngramPreprocess,
                ngram_range = (1, 5),
                stop_words = 'english',
                min_df = 2
            )),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(
                verbose = 2,
                class_weight = 'balanced',
            ))])
    clf.fit(corpus, labels)
    with open('pickles/ngramBinaryClf.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clf


'''
Given a binary classifier, tests it on a testing set of data.
Args:
    classifier (sklearn classifier): classifier to test
    path (string): path to directory of testing data
    func (method): method to use in readRaw method to extract relevant features
Returns:
    score (float): accuracy of classifier as a float between 1 and 0
'''
def testClassifier(classifier, path, func, skip = 1):
    testing, labels = readRaw(path, func, skipRate = skip)
    score = classifier.score(testing, labels)
    return score
