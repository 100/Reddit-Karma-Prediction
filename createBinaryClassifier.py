from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk import PorterStemmer
import os, json, string, math, random
try:
   import cPickle as pickle
except:
   import pickle

def readRaw():
    texts = []
    labels = []
    for corpusFile in os.listdir(os.getcwd()):
        if corpusFile.startswith('redditcorp') and corpusFile.endswith('.txt'):
            jsonFile = open(corpusFile).read()
            jsonBlobs = jsonFile.split('\n')
            random.shuffle(jsonBlobs)
            unpopular = 0
            for blob in jsonBlobs[::50]:
                try:
                    comment = json.loads(blob)
                except ValueError:
                    continue
                if (comment['score_hidden'] == False):
                    texts.append(comment['body'])
                    labels.append('popular' if comment['score'] > 0 else 'unpopular')
    return texts, labels

def createBinaryClassifier(corpus, labels):
    clf = Pipeline([
            ('vect', CountVectorizer(
                decode_error = 'replace',
                strip_accents = 'unicode',
                preprocessor = process,
                ngram_range = (2,2),
                stop_words = 'english',
                min_df = 2
            )),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                verbose = 2,
                class_weight = 'balanced',
                n_iter = 100
            ))])
    clf.fit(corpus, labels)
    with open('binaryClf.pkl', 'wb') as pickleFile:
        pickle.dump(clf, pickleFile)
    return clf

def process(comment):
    starting = str(comment.encode('utf-8','replace')).lower()
    starting = starting.replace('](', '] (')
    tokens = starting.split()
    acceptablePunct = string.punctuation.replace('<=>', '=')
    for idx, token in enumerate(tokens):
        tokens[idx] = ''.join(char for char in token if
        char not in acceptablePunct)
    acceptable = string.ascii_lowercase + string.digits
    tokens = [PorterStemmer().stem_word(token) for token in tokens if
            False not in [letter in acceptable for letter in token]
            and len(token) > 0]
    return ' '.join(tokens)

createBinaryClassifier(*readRaw())
