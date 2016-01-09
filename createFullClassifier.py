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


'''
For use with readRaw method. Extracts comment body and labels with appropriate
bin according to its score.
Args:
    comment (string): comment body
    texts (list of string): list to append current comment's body to
    labels (list of string): list to append current comment's classification to
'''
def seekKarma(comment, texts, labels):
    texts.append(comment['body'])
    labels.append(assignBin(comment['score']))


'''
For use in seekKarma method. Classifies a comment to one of five bins depending
on its score.
Args:
    score (int): score of comment to be classified
Returns:
    (string): name of bin to which comment was classified
'''
def assignBin(score):
    if score <= 0: return 'negative'
    if score >= 1 and score <= 3: return 'low'
    if score >= 4 and score <= 10: return 'medium'
    if score >= 11 and score <= 20: return 'high'
    if score >= 21: return 'very high'


'''
Given a comment body, convert it into its feature vector.
Args:
    blobber (Textblob Blobber): pre-trained Blobber object
    comment (string): comment body
    ngramClf (sklearn classifier): binary classifier trained on n-grams
Returns:
    vector (list): feature vector for this comment
'''
def vectorize(blobber, comment, ngramClf):
    cleansed = ngramPreprocess(comment, lemm = False)
    polarity, subjectivity, pos, neg = sentimentAnalysis(cleansed, blobber)
    with open('pickles/swearList.pkl', 'rb') as swearPickle:
        sentences, words, characters, averageWordLen, swears = counts(cleansed, pickle.load(swearPickle))
    vector = [sentences, words, characters, averageWordLen, swears, polarity,
                subjectivity, pos, neg,
                1 if ngramClf.predict([comment]) == ['popular'] else 0]
    return vector

def vectorizeNoNgram(blobber, comment):
    cleansed = ngramPreprocess(comment, lemm = False)
    polarity, subjectivity, pos, neg = sentimentAnalysis(cleansed, blobber)
    with open('pickles/swearList.pkl', 'rb') as swearPickle:
        sentences, words, characters, averageWordLen, swears = counts(cleansed, pickle.load(swearPickle))
    vector = [sentences, words, characters, averageWordLen, swears, polarity,
                subjectivity, pos, neg, 1]
    return vector


'''
Creates and trains classifier with metadata features, and pickles to disk
upon completion. Since mass-vectorizing comments is a relatively slow process,
includes verbosity (print current progress).
Args:
    blobber (Textblob Blobber): pre-trained Blobber object
    ngramClf (sklearn classifier): binary classifier trained on n-grams
    corpus (list of string): list of comment bodies
    labels (list of string): list of comment classifications
    verbose (boolean) [optional]: print vectorization status if true
Returns:
    classifier (sklearn classifier)
    matrix (list of lists): matrix representing vectorized training data
    labels (list of string): list of comment classifications
'''
def createFullClassifier(blobber, ngramClf, corpus, labels, verbose = True):
    matrix = []
    if verbose == True:
        count = 0
        for comment in corpus:
            matrix.append(vectorize(blobber, comment, ngramClf))
            count += 1
            print ('Vectorized ' + str(count) + ' data points out of ' +
                str(len(corpus)) + '.')
    else:
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
    with open('pickles/fullClassifier.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clf, matrix, labels


'''
Given a non-binary classifier, tests it on a testing set of data.
Args:
    blobber (Textblob Blobber): pre-trained Blobber object
    ngramClf (sklearn classifier): binary classifier trained on n-grams
    classifier (sklearn classifier): classifier to test
    path (string): path to directory of testing data
    func (method): method to use in readRaw method to extract relevant features
    skip (int) [optional]: skip rate for readRaw method
Returns:
    score (float): accuracy of classifier as a float between 1 and 0
'''
def testClassifier(blobber, ngramClf, classifier, path, func, skip = 1):
    testing, labels = readRaw(path, func, skipRate = skip)
    matrix = [vectorize(blobber, comment, ngramClf) for comment in testing]
    score = classifier.score(matrix, labels)
    return score


'''
Given a classifier, perform grid search to find optimal parameter values, and
prints the best values.
Args:
    clf (sklearn classifier): classifier to perform search with
    corpus (list of string): matrix of vectorized comment bodies
    labels (list of string): list of comment classifications
    parameters (dict): dictionary of {'parameter': (values,)} to test in the grid search
        Example: parameters = {'clf__C': (1, 10, .01, .001)}
Prints:
    Best values for each specified parameter
'''
def tuneParams(clf, corpus, labels, parameters):
    gridClf = GridSearchCV(clf, parameters, verbose = 2, pre_dispatch = 1)
    gridClf.fit(corpus, labels)
    bestParameters = gridClf.best_estimator_.get_params()
    print 'best score: ' + str(gridClf.best_score_)
    for paramName in sorted(parameters.keys()):
        print('\t%s: %r' % (paramName, bestParameters[paramName]))
