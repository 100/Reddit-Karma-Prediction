from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer
import requests
import string, os, sys, json, math, random
try:
   import cPickle as pickle
except:
   import pickle


'''
Assuming that data is stored as JSON in text files delimited by newlines,
iterates through all files in the given directory, and executes the function in
the parameter to record user-defined data.
Args:
    path (string) - relative file location of training data
    func (method) - operations to execute on each comment
    skipRate (int) [optional]: Given a larger than needed training set, only use
        every $skipRate comment in each file.
Returns:
    Contents of texts and labels (lists) from the processed dataset.
'''
def readRaw(path, func, skipRate = 1):
    texts = []
    labels = []
    for corpusFile in os.listdir(path):
        if corpusFile.startswith('redditcorp') and corpusFile.endswith('.txt'):
            jsonFile = open(os.path.join(path, corpusFile)).read()
            jsonBlobs = jsonFile.split('\n')
            random.shuffle(jsonBlobs)
            for blob in jsonBlobs[::skipRate]:
                try:
                    comment = json.loads(blob)
                except ValueError:
                    continue
                if (comment['score_hidden'] == False):
                    func(comment, texts, labels)
    return texts, labels


'''
Obtains a list of swear words and pickles it to disk.
Returns:
    List of swear words (strings)
'''
def getSwearList():
    raw = requests.get('http://www.cs.cmu.edu/~biglou/resources/bad-words.txt')
    swearList = [str(swear) for swear in raw.text.split()]
    with open('swearList.pkl', 'wb') as pickleFile:
        pickle.dump(swearList, pickleFile, pickle.HIGHEST_PROTOCOL)
    return swearList


'''
Preprocesses comment bodies for classification. Removes punctuation, converts to
lowercase, removes non-alphanumeric characters. Optionally lemmatizes comment.
Args:
    comment (string): body of comment to Preprocesses
    lemm (boolean) [optional]: indication of whether or not to lemmatize comment
Returns:
    Processed comment body (string)
'''
def ngramPreprocess(comment, lemm = True):
    starting = str(comment.encode('utf-8','replace')).lower()
    starting = starting.replace('](', '] (')
    tokens = starting.split()
    acceptablePunct = string.punctuation.replace('<=>', '=')
    for idx, token in enumerate(tokens):
        tokens[idx] = ''.join(char for char in token if
        char not in acceptablePunct)
    acceptable = string.ascii_lowercase + string.digits
    if (lemm):
        tokens = [Word(token).lemmatize() for token in tokens if
                    False not in [letter in acceptable for letter in token]
                    and len(token) > 0]
    else:
        tokens = [token for token in tokens if
                    False not in [letter in acceptable for letter in token]
                    and len(token) > 0]
    return ' '.join(tokens)


'''
Performs two types of sentiment analysis on a comment.
Args:
    comment (string): body of comment
    blobber (Blobber): Textblob Blobber trained on NaiveBayesAnalyzer
Returns:
    As tuple:
        polarity, subjectivity, pos, neg [int]: analysis results
'''
def sentimentAnalysis(comment, blobber):
    patternsSenti = TextBlob(comment)
    polarity = patternsSenti.sentiment.polarity
    subjectivity = patternsSenti.sentiment.subjectivity
    bayesSenti = blobber(comment)
    pos = bayesSenti.sentiment.p_pos
    neg = bayesSenti.sentiment.p_neg
    return polarity, subjectivity, pos, neg


'''
Extracts word, character, and sentence count as well as average lengths.
Args:
    comment (string): body of comment
    swearList (list of string): list of words to consider swears
Returns:
    As tuple:
        sentences, words, characters, swears [int],
        average word length [float]
'''
def counts(comment, swearList):
    blob = TextBlob(comment)
    sentences = len(blob.sentences)
    words = len(blob.words)
    if words > 0:
        characters = sum([len(word) for word in blob.words])
        averageWordLen = float(characters) / words
        swears = len(set(comment.split()) & set(swearList))
    else:
        characters, averageWordLen, swears = 0, 0, 0
    return sentences, words, characters, averageWordLen, swears
