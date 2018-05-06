import numpy as np

def mode2(a, axis=0, vals=False):
	scores = np.unique(np.ravel(a))
	testshape = list(a.shape)
	testshape[axis] = 1
	oldmostfreq = np.zeros(testshape)
	oldcounts = np.zeros(testshape)
	
	for score in scores:
		template = (a == score)
		counts = np.expand_dims(np.sum(template, axis), axis)
		mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
		oldcounts = np.maximum(counts, oldcounts)
		oldmostfreq = mostfrequent
	
	return oldcounts[0] if vals else mostfrequent[0]

##################################################################################
##################################################################################

def standardizeRDD(rdd, n_cols, types):
	rows = rdd.count()
	c_means = np.zeros(n_cols)
	c_devs = np.zeros(n_cols)
	for i in range(n_cols):
		if types[i] == float:
			col = rdd.map(lambda row: row[i]).filter(lambda val: val != '')
			c_means[i], c_devs[i] = col.mean(), col.stdev()

	rdd_std = rdd.map(lambda row: [(v-c_means[p])/c_devs[p] if (types[p]==float)&(v!='') else v for p,v in enumerate(row)])
	return rdd_std

#################################################################################
#################################################################################

def typeCol(col, n_rows):
	total = col.map(lambda row: isinstance(row, (int, float))).collect()
	return int(all(total))

def getMode(col):
	col_d = col.map(lambda row: (row, 1)).countByKey()
	return max(col_d.items(), key = lambda kv: kv[1])[0]

def getMedian(col_unsorted, n_rows):
	col = col_unsorted.sortBy(lambda val: val)
	
	median_index = int(n_rows/2) + 1
	type_int = isinstance(col.first(), int)
	
	if n_rows % 2 == 1:
		return col.take(median_index)[-1]
	else:
		medians = col.take(median_index)
		if type_int:
			return int((medians[-1] + medians[-2])/2)
		else:
			return (medians[-1] + medians[-2])/2	

def replaceNA(rdd, n_cols):
	n_rows = rdd.count()
	reps = [0 for x in range(n_cols)]

	for i in range(n_cols):
		col = rdd.map(lambda row: row[i]).filter(lambda x: x != '')
		col_n = col.count()
		if col_n == n_rows:
			continue
		col_type = typeCol(col, col_n)
		if col_type == 1.0:
			reps[i] = getMedian(col, col_n)
		else:
			reps[i] = getMode(col)
	rdd_c = rdd.map(lambda row: [val if val != '' else reps[pos] for pos,val in enumerate(row)])
	return rdd_c


################################################################################
################################################################################

def hasNA(rdd):
	return rdd.zipWithIndex().filter(lambda lineIndex: '' in lineIndex[0]).map(lambda lineIndex: lineIndex[1]).collect()

def rmNA(rdd):
	return rdd.filter(lambda line: '' not in line)

################################################################################
################################################################################
def floatable(n):
	if n.lstrip('-').isdigit():
		return int(n)
	try: 	
		return round(float(n), 1)
	except:
		return n

def prepRDD(sc, path, splitter, rm_na, rep_na):
	rdd = sc.textFile(path).map(lambda line: line.split(splitter))

	header = rdd.first()
	rdd_data = rdd.filter(lambda line: line != header).map(lambda line: [floatable(el) for el in line])
	types = [all(rdd_data.map(lambda row: row[i]).filter(lambda v: v != '').map(lambda v: isinstance(v, (int, float))).collect()) for i in range(len(header))]
	
	if rm_na:
		return (header, types), rmNA(rdd_data)
	elif rep_na:
		return (header, types), replaceNA(rdd_data, len(header))
	else:
		return (header, types), rdd_data

#################################################################################
#################################################################################

def typeCol(col, n_rows):
	total = col.map(lambda row: isinstance(row, (int, float))).collect()
	return int(all(total))

def getMode(col):
	col_d = col.map(lambda row: (row, 1)).countByKey()
	return max(col_d.items(), key = lambda kv: kv[1])[0]

def getMedian(col_unsorted, n_rows):
	col = col_unsorted.sortBy(lambda val: val)
	
	median_index = int(n_rows/2) + 1
	type_int = isinstance(col.first(), int)
	
	if n_rows % 2 == 1:
		return col.take(median_index)[-1]
	else:
		medians = col.take(median_index)
		if type_int:
			return int((medians[-1] + medians[-2])/2)
		else:
			return (medians[-1] + medians[-2])/2	

def replaceNA(rdd, n_cols):
	n_rows = rdd.count()
	reps = [0 for x in range(n_cols)]

	for i in range(n_cols):
		col = rdd.map(lambda row: row[i]).filter(lambda x: x != '')
		col_n = col.count()
		if col_n == n_rows:
			continue
		col_type = typeCol(col, col_n)
		if col_type == 1.0:
			reps[i] = getMedian(col, col_n)
		else:
			reps[i] = getMode(col)
	rdd_c = rdd.map(lambda row: [val if val != '' else reps[pos] for pos,val in enumerate(row)])
	return rdd_c


################################################################################
################################################################################

def hasNA(rdd):
	return rdd.zipWithIndex().filter(lambda lineIndex: '' in lineIndex[0]).map(lambda lineIndex: lineIndex[1]).collect()

def rmNA(rdd):
	return rdd.filter(lambda line: '' not in line)

################################################################################
################################################################################
def floatable(n):
	if n.lstrip('-').isdigit():
		return int(n)
	try: 	
		return round(float(n), 1)
	except:
		return n

def prepRDD(sc, path, splitter, rm_na, rep_na):
	rdd = sc.textFile(path).map(lambda line: line.split(splitter))
	
	header = rdd.first()
	rdd_data = rdd.filter(lambda line: line != header).map(lambda line: [floatable(el) for el in line])

	info = {}	
	info['n_rows'] = rdd_data.count()
	info['n_ftrs'] = len(header)
	info['head'] = header
	info['types'] = [float if all(rdd_data.map(lambda row: row[i]).filter(lambda v: v != '').map(lambda v: isinstance(v, (int, float))).collect()) else str for i in range(len(header))]
	
	if rm_na:
		return info, rmNA(rdd_data)
	elif rep_na:
		return info, replaceNA(rdd_data, len(header))
	else:
		return info, rdd_data
