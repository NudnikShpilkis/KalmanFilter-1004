

def typeCol(col, n_rows):
    total = col.map(lambda row: isinstance(row, (int, float))).sum()
    return total/n_rows

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

def prepRDD(rdd, splitter, rm_na, rep_na):
    rdd_s = rdd.map(lambda line: line.split(splitter))

    header = rdd_s.first()
    rdd_data = rdd_s.filter(lambda line: line != header).map(lambda line: [floatable(el) for el in line])

    if rm_na:
        return header, rmNA(rdd_data)
    elif rep_na:
        return header, replaceNA(rdd_data, len(header))
    else:
        return header, rdd_data
