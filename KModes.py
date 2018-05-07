import numpy as np
import itertools
from collections import Counter
from utils import *

class Feature:
#Feature subclass for better use with KModes
    def __init__(self, featType, values):
        #values in array form and feature dtype
        self.vals = np.array(values.filter(lambda v: v != '').collect())
        self.missing = np.array(values.zipWithIndex().filter(lambda k_i: k_i[0] == '').map(lambda k_i: k_i[1]).collect())
        self.size = self.vals.shape[0]
        self.type = featType
        #unique values, their orig indices and counts
        vals_unique, inds, counts = np.unique(self.vals, return_inverse=True, return_counts=True)
        #Feature dictionary for use in itialization
        self.valDict = {v:c for v,c in  zip(vals_unique, counts)}
        #frequency of values in orig indices
        self.valFreq = counts[inds]
        self.valFreqMiss = self.fillMissing(self.valFreq)
        return
    
    def fillMissing(self, arr):
        for miss in self.missing:
            arr = np.concatenate((arr[:miss], [np.nan], arr[miss:]))
        return arr
    
    def getCenter(self, cMsk):
    #Get center of feature in given centroid
        #Mean for num features
        msk = cMsk[np.setdiff1d(np.arange(cMsk.shape[0]), self.missing)]
        if self.type==float:
            return self.vals[msk].mean()
        #Mode for cat features
        else:
            return mode2(self.vals[msk], vals=False)        
    
    def getDissimilarity(self, cntr, cMsk=None):
    #Given centroid center, get dissimilarity
        #Numerical feature use euclidean distance
        if self.type==float: 
            dissims = (self.vals - cntr)**2
        else:
            #Get Weight of points, from WHICH paper?
            weights = (self.valFreq + self.valDict[cntr])/(self.valFreq * self.valDict[cntr])
            #2007 dissimilarity
            if cMsk is not None:
                msk = cMsk[np.setdiff1d(np.arange(cMsk.shape[0]), self.missing)]
                dissims = weights * (1-(self.vals==cntr).astype(int) *((self.vals[msk] == cntr).sum()/msk.sum()))
            #Classic dissimilarity
            else:
                dissims = weights * (self.vals != cntr).astype(int)
                
    dissims = self.fillMissing(dissims, high).astype(np.float64)
    return dissims
                
    def calcWeightedEntropy(self, cMsk):
        msk = cMsk[np.setdiff1d(np.arange(cMsk.shape[0]), self.missing)]
        local_vals = self.vals[msk]
        if local_vals.shape[0] == 1: return 0
        
        keys, counts_raw = np.unique(local_vals, return_counts=True)        
        counts = counts_raw[counts_raw.nonzero()]
        probs = counts/counts.sum() 

        local_entropy = -np.sum(probs * np.log(probs)) 
        entropy_weight = 2*(1 - 1/(1 + np.exp(-local_entropy)))
        return entropy_weight * local_entropy

#######################################################################################################################

def initCentroids(features, info, weights):
#Init centroids based on Cao Paper
    #Empty list of centroids
    cntrds = [0 for i in range(info['n_clstrs'])]
    #Density of each point
    dens = np.zeros(info['n_rows'])
    #Iterate through features
    for feat in features:
        #Update point density
        dens += (feat.valFreqMiss/(info['n_rows']*info['n_clstrs']))
    #cntrds[0] = [feat.vals[dens.argmax()] for feat in features]
    cntrds[0] = [feat.vals[np.nanargmax(dens)] for feat in features]
    #Iterate though remaining centroids, adapted from github implementation 
    for ki in range(1, info['n_clstrs']):
        adj_dens = np.zeros((ki, info['n_rows']))
        #Iterate through already created centroids
        for kii in range(0,ki):
            #Similarity btwn cntrds and pts
            dissims = np.zeros((info['n_rows'], info['n_ftrs']))
            #dissims = np.zeros(info['n_rows'])
            #Iterate through features
            for f, feat in enumerate(features):
                dissims[:,f] += feat.getDissimilarity(cntrds[kii][f], high=False)
            #Densities * similarites
            adj_dens[kii] = (dens * (weights * dissims.sum(axis=1))).astype(np.float64)
        #New centroid has max dens*sims from previous centroids
        cntrds[ki] = [feat.vals[np.nanargmax(np.nanmin(adj_dens.astype(np.float64), axis=0))] for feat in features]
    #Assign intial membship
    cntrds, membship, cost = assignMembship(cntrds, features, info, weights)
    return cntrds, membship, cost

def assignMembship(cntrds, features, info, weights, membship=None, pred=True):
#Assign membship to centroids, membship=True if dissim is 2007, pred=True for KModes predict
    dissims = np.zeros((info['n_rows'], info['n_clstrs']))
    #Iterate through centroids
    for i, cntrd in enumerate(cntrds):
        #Get num and cat feat sims  
        num_feats = np.zeros((info['n_rows'], info['num']))
        cat_feats = np.zeros((info['n_rows'], info['cat']))
        n, c = 0, 0
        #Iterate thorugh features
        for f, feat in enumerate(features):
            #Num features
            if isinstance(feat.type, (float, int)):
                num_feats[:,n] += feat.getDissimilarity(cntrd[f], high=True)
                n += 1
            #Cat geatures
            else:
                #Choose which dissim to use
                if membship is not None:
                    #2007 dissim
                    cntrdMask = (membship == i)
                    cat_feats[:,c] += feat.getDissimilarity(cntrd[f], high=True, cMsk=cntrdMask)
                else:
                    #Orig disim
                    cat_feats[:,c] += feat.getDissimilarity(cntrd[f], high=True)
                c += 1
        #Sims of given centroid
        dissims[:,i] = weights * np.nansum(num_feats, axis=1) + info['gamma'] * weights * np.nansum(cat_feats, axis=1)
    #Choose smallest dissim for each pt
    membship = np.nanargmin(dissims, axis=1)
    #Cost is sum of smallest dissim
    cost = np.nanmin(dissims, axis=1).sum()
    #If centroid has no assigned pts
    if (np.unique(membship).shape[0] != info['n_clstrs'])&(not pred):
        print('meep meep')
        cntrds = newCentroids(cntrds, features, membship, info)
        #Recursively assign membship
        cntrds, membship, cost = assignMemship(cntrds, features, info, weights, membship)
    return cntrds, membship, cost

def newCentroids(cntrds, features, membship, info):
#Create new centroids if centroid is not present in membship
    #Missing centroids
    missing = np.setdiff1d(range(0,info['n_clstrs']), membship)
    #Largest centroid
    max_cntrd = mode2(membship, vals=False)
    #Choose pts from largest centroid
    new_cntrds = np.random.choice(np.where(membship == max_cntrd)[0], missing.shape[0])
    #Replace old centroids with new points
    for m, nc in zip(missing, new_cntrds):
        cntrds[m] = [feat.vals[new_cntrds[nc]] for feat in features]
    return cntrds

def setCentroids(features, membship, info, weights, dissim):
#Set centroids from center of each feature
    cntrds = [0 for c in range(info['n_clstrs'])]
    #Iterate through centroids, get approp center for each feature
    for i in range(info['n_clstrs']):
        cntrds[i] = [feat.getCenter(membship==i) for feat in features]
    #2007 dissim, pass membship
    if dissim:
        cntrds, membship, cost = assignMembship(cntrds, features, info, weights, membship)
    #Classsic dissim
    else:
        cntrds, membship, cost = assignMembship(cntrds, features, info, weights)
    return cntrds, membship, cost
 
def setMembshipToMatrix(sc, membship, info):
#Convert membship to 2D array
    return sc.parallelize([[1 if membship[pos] else 0 for i in range(info['n_ftrs'])] for pos in range(info['n_rows'])])

def doGWE(features, msk, weights):
    assigned = np.where(msk)[0]
    tot_e = np.sum([weights*feat.calcWeightedEntropy(msk) for feat in features]) #calc initial weighted-entropy
    o_factors = np.array([tot_e for x in range(assigned.shape[0])])
    for o, i in enumerate(assigned):
        msk[i] = False
        wo = np.sum([weights*feat.calcWeightedEntropy(msk) for feat in features])
        o_factors[o]  -= wo if wo != 0 else (tot_e + 1)
        msk[i] = True
    msk_msk = (o_factors > 0) 
    pts = np.zeros(msk.shape[0], dtype=bool)
    pts[assigned] = msk_msk
    return np.arange(msk.shape[0])[pts]

def getOutliers(features, info, membship, weights):
    outliersList =  [0 for i in range(info['n_clstrs'])]
    
    for i in range(info['n_clstrs']):
        outliersList[i] = doGWE(features, membship == i, weights)
       
    outliers = np.array(list(itertools.chain.from_iterable(outliersList)))  
    
    return np.sort(outliers)


#######################################################################################################################

class KModes(BaseEstimator, ClusterMixin):
#Implementation of KModes, with help from github implementation
    def __init__(self, n_clusters=5, max_iter=100, gamma=1, std=True, dissim=False):
        #Keep all size information in dictionary
        self.info = {}
        self.info['n_clstrs'], self.info['max_itr'] = n_clusters, max_iter
        self.info['gamma'] = 1
        #2007 dissim or classic dissim
        self.std, self.dissim = std, dissim
        self.centroids_, self.membship_, self.cost_, self.itr_, self.labels_ = None, None, None, None, None
        return     
    
    def fit(self, sc, rdd, info):
    #Fit KModes to data
        for k,v in info.items():
            self.info[k] = v
        assert self.info['n_clstrs'] <= self.info['n_rows'], 'More Clusters than rows'

        if self.std == True:
            rdd = standardizeRDD(rdd, info['n_ftrs'], info['types'])
        
        #Features to better form
        self.features_ = np.array([Feature(self.info['types'][i], rdd.map(lambda row: row[i])) for i in range(self.info['n_ftrs'])])
        missing = Counter(np.array(list(itertools.chain.from_iterable([feat.missing for feat in self.features_]))))
        self.weights_ = np.array([self.info['n_rows']/(self.info['n_rows']-missing[k]) if k in missing else 1 for k in range(self.info['n_rows'])])
        #Save n_ftrs and c_ftrs
        self.info['num'] = int(np.sum([1 for feat in self.features if isinstance(feat.type, (float, int))]))
        self.info['cat'] = int(self.info['n_ftrs'] - self.info['num'])
        
        #Init centroids, membship and cost
        cntrds, membship, cost = initCentroids(self.features_, self.info, self.weights_)
    
        #Iterate until convergence or max iteration reached
        itr, converged = 0, False
        while (itr <= self.info['max_itr'])&(not converged):
            itr += 1
            #Set centroids, membship
            cntrds, new_membship, new_cost = setCentroids(self.features_, membship, self.info, self.weights_, self.dissim)
            #Check if any pts changed membship
            chngs = np.sum(new_membship != membship)
            #Convergence conditions
            converged = (chngs == 0)|(new_cost >= cost)
            membship, cost = new_membship, new_cost
        
        #Save fit info
        self.outliers_ =getOutliers(self.features_, self.info, membship, self.weights_)
        self.membship_ = membship
        self.matrix_ = setMembshipToMatrix(sc, membship, self.info)
        self.centroids_, self.cost_, self.itr_ = cntrds, cost, itr
        return self   
    
    def predict(self, sc, rdd, info):
    #Given new data, assignMembship to centroids
        if membship is None: return "Fit before predict"
        if (info['types'] != self.info['types'])|(info['n_ftrs'] != self.info['n_ftrs']): return 'New data does not match old data'
        
        new_features = np.array([Feature(rdd.map(lambda row: row[i]), info['types'][i]) for i in range(info['n_ftrs'])])
        new_missing = Counter(np.array(list(itertools.chain.from_iterable([feat.missing for feat in new_features]))))
        new_weights = np.array([info['n_rows']/(info['n_rows']-new_missing[k]) if k in new_missing else 1 for k in range(info['n_rows'])])
        new_info = self.info.copy()
        new_info['n_rows'] = info['n_rows']
        cntrds, membship, cost = assignMembship(self.centroids_, new_features, new_info, new_weights, pred=True)
        new_mat = setMembshipToMatrix(sc, membship, new_info)
        return new_mat
    
    def fit_predict(self, sc, rdd, info):
        return self.fit(sc, rdd, info).predict(sc, rdd, info)
