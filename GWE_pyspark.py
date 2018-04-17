import sys
import numpy as np

def calcEntropy(labels):	
	counts_raw = np.array(list(labels.values()))
	if counts_raw.shape[0] == 1:
		return 0
	counts = counts_raw[counts_raw.nonzero()]
	probs = counts/counts.sum()

	feature_entropy = -np.sum(probs * np.log(probs))
	weight  = 2 * (1- 1/(1 + np.exp(-feature_entropy)))
	return weight * feature_entropy

def rowEntropy(row, dicts):
	entropy_row = np.zeros(len(dicts))
	for pos, key in enumerate(row):
		if key == '':
			entropy_row[pos] = calcEntropy(dicts[pos])
		else:
			dicts[pos][key] -= 1
			entropy_row[pos] = calcEntropy(dicts[pos])
			dicts[pos][key] += 1
	
	return entropy_row

def GWE(rdd, cols):
	d1 = [rdd.map(lambda line: (line[i], i)).filter(lambda kv: kv[0] != '').countByKey() for i in range(cols)]
	tot_e = np.sum([calcEntropy(d) for d in d1])
	o_factors_index = rdd.zipWithIndex().map(lambda row: (tot_e - rowEntropy(row[0], d1).sum(), row[1]))
	outliers = o_factors_index.filter(lambda row: row[0] > 0).map(lambda row: row[1]).collect()
	o_factors = o_factors_index.map(lambda row: row[0]).collect()
	
	return o_factors, outliers

###########################################
###########################################
###########################################


