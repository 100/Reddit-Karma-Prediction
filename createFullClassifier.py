from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import numpy
import math
try:
   import cPickle as pickle
except:
   import pickle
from preprocessing import ngramPreprocess, sentimentAnalysis, counts, readRaw


def seekKarma(comment, texts, labels):
    texts.append(comment['body'])
    labels.append(assignBin(comment['score']))

def assignBin(score):
    if score <= 0: return 'negative'
    if score >= 1 and score <= 5: return 'low'
    if score >= 6 and score <= 15: return 'medium'
    if score >= 16 and score <= 50: return 'high'
    if score >= 51: return 'very high'

def vectorize(blobber, comment, ngramClf):
    cleansed = ngramPreprocess(comment, lemm = False)
    polarity, subjectivity, pos, neg = sentimentAnalysis(cleansed, blobber)
    with open('swearList.pkl', 'rb') as swearPickle:
        sentences, words, characters, averageWordLen, swears = counts(cleansed, pickle.load(swearPickle))
    vector = [sentences, words, characters, averageWordLen, swears, polarity,
                subjectivity, pos, neg,
                1 if ngramClf.predict([comment]) == ['popular'] else 0]
    return vector

def createFullClassifier(blobber, ngramClf, corpus, labels):
    matrix = []
    for comment in corpus:
        matrix.append(vectorize(blobber, comment, ngramClf))
    clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(
                verbose = 2,
                class_weight = 'balanced',
                max_iter = 1000,
                C = .001
            ))])
    clf.fit(numpy.array(matrix), labels)
    with open('fullClassifier.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clf, matrix, labels

def testClassifier(blobber, ngramClf, classifier, path, func, skip):
    testing, labels = readRaw(path, func, skip)
    matrix = [vectorize(blobber, comment, ngramClf) for comment in testing]
    score = classifier.score(matrix, labels)
    return score

def tuneParams(clf, corpus, labels):
    parameters = {'clf__C': (1, 10, .01, .001)}
    gridClf = GridSearchCV(clf, parameters, verbose = 2, pre_dispatch = 1)
    gridClf.fit(corpus, labels)
    bestParameters = gridClf.best_estimator_.get_params()
    print 'best score: ' + str(gridClf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, bestParameters[param_name]))
