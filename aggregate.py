from sklearn.cluster import KMeans
import numpy
import os, random, json, operator, math
try:
   import cPickle as pickle
except:
   import pickle


'''
Compute average karma per comment in each subreddit present in the training set,
and save resulting dictionary to disk. May be very time-consuming, so includes
verbosity option.
Args:
    path (string): file path to directory of training data
    skipRate (int) [optional]: use every $skipRate comment in the training data
    top (int) [optional]: only use the top $top percent of subreddits ordered by
        total number of comments
    verbose (boolean) [optional]: print status of current progress
Returns:
    subredditAverages (dict): {subreddit: average karma} pairs
'''
def computeAverages(path, skipRate = 1, top = 10, verbose = True):
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
                if verbose == True:
                    print 'Completed ' + str(numComments) + ' comments.'
    print str(totalKarma) + ' total karma across all comments.'
    print str(numComments) + ' comments analyzed.'
    subredditAverages = {}
    for subreddit, score in subredditScores.iteritems():
        subredditAverages[subreddit]= float(score) / subredditNumComs[subreddit]
    sortedNumComs = sorted(subredditNumComs.items(),
        key = operator.itemgetter(1))
    cutoff = int((len(subredditNumComs)) * (float(top)/100))
    cutoffNumComs = sortedNumComs[-cutoff:]
    subredditAverages = dict([(subreddit[0], subredditAverages[subreddit[0]])
        for subreddit in cutoffNumComs])
    with open('pickles/subredditAverages.pkl', 'wb') as pickleFile:
        pickle.dump(subredditAverages, pickleFile, pickle.HIGHEST_PROTOCOL)
    with open('pickles/cutoffNumComs.pkl', 'wb') as pickleFile:
        pickle.dump(dict(cutoffNumComs), pickleFile, pickle.HIGHEST_PROTOCOL)
    return dict(cutoffNumComs), subredditAverages


'''
Creates dict representation of data needed for cytoscape.js graph
representation, and saves to disk.
Args:
    edgesPerNodes (int) [optional]: $edgesPerNode edges in each subreddit node to each other;
        provide -1 to create densely-connected clusters
Returns:
    clusterDict (dict): JSON representation of data needed for cytoscape.js
'''
def createClusters(edgesPerNode = 0):
    with open('pickles/subredditAverages.pkl', 'rb') as pickleFile:
        averagesDict = pickle.load(pickleFile)
    with open('pickles/cutoffNumComs.pkl', 'rb') as pickleFile:
        numComs = pickle.load(pickleFile)
    kmeans = KMeans(
        n_clusters = 5,
        verbose = 2
    )
    kmeans.fit(numpy.array(averagesDict.values()).reshape(-1, 1))
    elements = []
    idCounter = 1
    for subreddit, avg in averagesDict.iteritems():
        elements.append({
            'data': {
                'id': idCounter,
                'sub': subreddit,
                'average': round(avg, 3),
                'cluster': numpy.asscalar(numpy.array([kmeans.predict([avg])[0]])),
                'size': int(15 * math.log(numComs[subreddit], 1000))
            },
            'position': {
              'x': numpy.asscalar(numpy.array([kmeans.predict([avg])[0]]))*400 + random.randint(-250, 250),
              'y': random.randint(0, 700)
        }})
        idCounter += 1
    clusterList = [[] for num in xrange(0, 5)]
    for subreddit in elements:
        id = subreddit['data']['id']
        cluster = subreddit['data']['cluster']
        clusterList[cluster].append(id)
    for cluster in clusterList:
        for subreddit in cluster:
            if edgesPerNode != -1:
                targets = [random.randint(0, len(cluster) - 1) for num in
                    xrange(edgesPerNode)]
            else:
                targets = [num for num in xrange(len(cluster))]
            for target in targets:
                elements.append({'data': {
                    'source': subreddit,
                    'target': cluster[target]
                }})
    for idx, cluster in enumerate(clusterList):
        elements.append({
            'data': {
                'id': 'cluster' + str(idx),
                'sub': 'cluster' + str(idx),
                'average': numpy.asscalar(numpy.array(kmeans.cluster_centers_[idx][0])),
                'cluster': idx,
                'size': 35
            },
            'position': {
              'x': idx*400 + random.randint(-50,50),
              'y': random.randint(300, 400)
        }})
        for sub in cluster:
            elements.append({'data': {
                'source': 'cluster' + str(idx),
                'target': sub
            }})
    with open('pickles/elements.pkl', 'wb') as pickleFile:
        pickle.dump(elements, pickleFile, pickle.HIGHEST_PROTOCOL)
    return elements
