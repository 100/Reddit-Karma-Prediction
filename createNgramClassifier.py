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


def seekPopular(comment, texts, labels):
    texts.append(comment['body'])
    labels.append('popular' if comment['score'] > 0 else 'unpopular')


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
    with open('ngramBinaryClf.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clf


def testClassifier(classifier, path, func):
    testing, labels = readRaw(path, func, 200)
    score = classifier.score(testing, labels)
    return score
