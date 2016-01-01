from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer
import requests
import string

def getSwearList():
    raw = requests.get('http://www.cs.cmu.edu/~biglou/resources/bad-words.txt')
    swearList = [str(swear) for swear in raw.text.split()]
    with open('swearList.pkl', 'wb') as pickleFile:
        pickle.dump(swearList, pickleFile, pickle.HIGHEST_PROTOCOL)
    return swearList

def ngramPreprocess(comment):
    starting = str(comment.encode('utf-8','replace')).lower()
    starting = starting.replace('](', '] (')
    tokens = starting.split()
    acceptablePunct = string.punctuation.replace('<=>', '=')
    for idx, token in enumerate(tokens):
        tokens[idx] = ''.join(char for char in token if
        char not in acceptablePunct)
    acceptable = string.ascii_lowercase + string.digits
    tokens = [Word(token).lemmatize() for token in tokens if
                False not in [letter in acceptable for letter in token]
                and len(token) > 0]
    return ' '.join(tokens)

#blobber needed to not re-train for each comment
#e.g. blobber = Blobber(analyzer = NaiveBayesAnalyzer())
def sentimentAnalysis(comment, blobber):
    patternsSenti = TextBlob(comment)
    polarity = patternsSenti.sentiment.polarity
    subjectivity = patternsSenti.sentiment.subjectivity
    bayesSenti = blobber(comment)
    pos = blob.sentiment.p_pos
    neg = blob.sentiment.p_neg
    return polarity, subjectivity, pos, neg

def counts(comment, swearList):
    blob = TextBlob(comment)
    sentences = len(blob.sentences)
    words = len(blob.words)
    characters = sum([len(comment) for word in words])
    averageWordLen = float(characters) / words
    swears = len(set(comment.split()) & set(swearList))
    return sentences, words, characters, averageWordLen, swears
