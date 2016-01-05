from sklearn.cluster import KMeans
import numpy
import os, random, json
try:
   import cPickle as pickle
except:
   import pickle


'''
Compute average karma per comment in each subreddit present in the training set,
and save resulting dictionary to disk.
Args:
    path (string): file path to directory of training data
    skipRate (int) [optional]: use every $skipRate comment in the training data
    minimum (int) [optional]: subreddit must have $minimum comments in total to
        be included in final dictionary
Returns:
    subredditAverages (dict): {subreddit: average karma} pairs
'''
def computeAverages(path, skipRate = 1, minimum = 25):
    subredditScores = {}
    subredditNumComs = {}
    totalKarma = 0
    numComments = 0
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
                if (comment['score_hidden']==False and 'subreddit' in comment):
                    if comment['subreddit'] in subredditScores:
                        subredditNumComs[comment['subreddit']] += 1
                        subredditScores[comment['subreddit']]+= comment['score']
                    else:
                        subredditNumComs[comment['subreddit']] = 1
                        subredditScores[comment['subreddit']] = comment['score']
                    totalKarma += comment['score']
                    numComments += 1
                print 'Completed ' + str(numComments) + ' comments.'
    print str(totalKarma) + ' total karma across all comments.'
    print str(numComments) + ' comments analyzed.'
    subredditAverages = {}
    for subreddit, score in subredditScores.iteritems():
        if subredditNumComs[subreddit] > 25:
            subredditAverages[subreddit]= float(score) /
                subredditNumComs[subreddit]
    with open('pickles/subredditAverages.pkl', 'wb') as pickleFile:
        pickle.dump(subredditAverages, pickleFile, pickle.HIGHEST_PROTOCOL)
    return subredditAverages


'''
Creates dict representation of data needed for alchemy.js graph representation,
and saves to disk.
Args:
    skipRate (int) [optional]: only use every $skipRate subreddit
    edgesPerNodes (int) [optional]: $edgesPerNode edges in each subreddit node
Returns:
    clusterDict (dict): JSON representation of data needed for alchemy.js
'''
def createClusters(skipRate = 15, edgesPerNode = 3):
    with open('pickles/subredditAverages.pkl', 'rb') as pickleFile:
        averagesDict = pickle.load(pickleFile)
    kmeans = KMeans(
        n_clusters = 5,
        verbose = 2
    )
    kmeans.fit(numpy.array(averagesDict.values()).reshape(-1, 1))
    clusterDict = {'nodes':[]}
    idCounter = 1
    for subreddit, avg in averagesDict.iteritems():
        clusterDict['nodes'].append({
            'id': idCounter,
            'sub': subreddit,
            'average': avg,
            'cluster': kmeans.predict([avg])[0]
        })
        idCounter += 1
    clusterDict['nodes'] = clusterDict['nodes'][::skipRate]
    clusterList = [[] for num in xrange(0, 5)]
    for subreddit in clusterDict['nodes']:
        id = subreddit['id']
        cluster = subreddit['cluster']
        clusterList[cluster].append(id)
    clusterDict['edges'] = []
    for cluster in clusterList:
        for subreddit in cluster:
            targets = [random.randint(0, len(cluster) - 1) for num in
                xrange(edgesPerNode)]
            for target in targets:
                clusterDict['edges'].append({
                    'source': subreddit,
                    'target': cluster[target]
                })
    with open('pickles/nodesDictLarge.pkl', 'wb') as pickleFile:
        pickle.dump(clusterDict, pickleFile, pickle.HIGHEST_PROTOCOL)
    return clusterDict
