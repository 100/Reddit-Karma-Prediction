from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import os, sys, json, math, random
try:
   import cPickle as pickle
except:
   import pickle
from preprocessing import ngramPreprocess


def readRaw(path):
    texts = []
    labels = []
    for corpusFile in os.listdir(path):
        if corpusFile.startswith('redditcorp') and corpusFile.endswith('.txt'):
            jsonFile = open(os.path.join(path, corpusFile)).read()
            jsonBlobs = jsonFile.split('\n')
            random.shuffle(jsonBlobs)
            for blob in jsonBlobs[::120]:
                try:
                    comment = json.loads(blob)
                except ValueError:
                    continue
                if (comment['score_hidden'] == False):
                    texts.append(comment['body'])
                    labels.append('popular' if comment['score'] > 0 else 'unpopular')
    return texts, labels


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
            ('clf', SGDClassifier(
                verbose = 2,
                class_weight = 'balanced',
                n_iter = 20
            ))])
    clf.fit(corpus, labels)
    with open('ngramBinaryClf.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clf


def testClassifier(classifier, path):
    testing, labels = readRaw(path)
    score = classifier.score(testing, labels)
    return score
