import numpy as np
    
    
# Takes a numpy array and returns the unique row values with counts
# Effectively a wrapper for numpy's unique function with specific settings
def attribute_partition(array):
    return np.unique(array, axis = 0, return_counts = True)

#Take list of counts for a partition of a set and returns the partition entropy
def calc_partition_entropy(partition_counts):
    total = np.sum(partition_counts)
    scaled = partition_counts/total
    return np.dot(-1*scaled, np.log2(scaled))

def calc_attribute_significance(X, attribute_index):
    full_partitions, full_counts = attribute_partition(X)
    removal_partitions, removal_counts = attribute_partition(np.concat(X[:,:attribute_index],
                                                                       X[:,attribute_index + 1:]))
    full_PE = calc_partition_entropy(full_counts)
    removal_PE = calc_partition_entropy(removal_PE)
    
    return (full_PE - removal_PE)/(full_PE + removal_PE)

def calc_attribute_weights(X):
    weights = {}
    num_zero = 0
    PE_zero = []
    num_attr = X.shape[1]
    
    for col_index in range(num_attr):
        sig = calc_attribute_significance(X, col_index)
        if sig > 0:
            weights[col_index] = 1 + sig
        else:
            num_zero += 1
            PE_zero.append(col_index)
         
    if num_zero > 0:
        zero_weight = (num_zero)/(num_attr + np.sqrt(num_attr - num_zero))
        for z_index in PE_zero:
            weights[z_index] = zero_weight
    
    return weights


def calc_object_significances(full_PE,part_counts):
    total = sum(part_counts)
    part_entropies = part_counts * np.log2(part_counts)
    total_part_ents = sum(part_entropies)
    partition_significance = []
    
    for index, c in enumerate(part_counts):
        part_removed_ent = (total - c) + (total_part_ents - part_entropies[index])/(total - c)
        
        if part_removed_ent > full_PE:
            partition_significance.append((part_removed_ent - full_PE)/(full_PE*c))
        else:
            partition_significance.append(0)
    
    return partition_significance                             

def peof(X):
    num_rows = X.shape[0]
    peof_factors = np.zeros(X.shape)
    sorted_attribute_weights = calc_attribute_weights(X)
    
    for j in range(len(sorted_attribute_weights)):
        w_j = sorted_attribute_weights[j]
        higher_attrs = sorted_attribute_weights[j+1:]
        W_j = sum(higher_attrs)/len(higher_attrs)
        
        attr_part, attr_counts = attribute_partition(X[:,j])
        attr_PE = calc_partition_entropy(attr_part)
        attr_partition_sigs = calc_object_significances(attr_PE, attr_counts)
 
        higher_attrs_part, higher_attrs_counts = attribute_partition(X[:,j+1:])
        higher_attrs_PE = calc_partition_entropy(higher_attrs_PE)
        higher_attrs_partition_sigs = calc_object_significances(higher_attrs_PE, higher_attrs_counts)

        for i in range(num_rows):
            row = X[i]
            attr_val = row[j]
            higher_attrs_val = row[j+1:]
            
            attr_partition = attr_part.index(attr_val)
            attr_cnt = attr_counts[attr_partition]
            attr_sig = attr_partition_sigs[attr_partition]
            
            higher_attrs_partition = higher_attrs_part.index(higher_attrs_val)
            higher_attrs_cnt = higher_attrs_counts[higher_attrs_partition]
            higher_attrs_sig = higher_attrs_partition_sigs[higher_attrs_partition]
            
            peof_factors[i][j] = (w_j * attr_cnt * attr_sig + W_j * higher_attrs_cnt * higher_attrs_sig)/(2*num_rows)
            
        return np.sum(peof_factors, axis =1)

            
        
        
###############################################################################
'''
def calcPartitionEntropy(counts):
    probs = counts/counts.sum()
    return -np.sum(probs * np.log(probs))

def calcSignificance(tot_pe, new_pe):
    return (tot_pe - new_pe)/(tot_pe + new_pe)

def calcPartitionEntropyAndSignificance(tot_pe, features):
    counts = np.array(list(itertools.chain.from_iterable([list(feat.valDict.values()) for feat in features])))
    probs = counts/counts.sum()
    part_pe = -np.sum(probs * np.log(probs))
    return (tot_pe - part_pe)/(tot_pe + part_pe)

def calcWeights(features, info):
    weights = {}
    c_zero = 0
    zero_ind = np.zeros(features.shape[0], dtype=int)
    tot_counts = np.array(list(itertools.chain.from_iterable([list(feat.valDict.values()) for feat in features])))
    tot_pe = calcPartitionEntropy(tot_counts)
    
    for f, featRem in enumerate(features):
        sig = calcPartitionEntropyAndSignificance(tot_pe, [feat for feat in features if feat != featRem])
        
        if sig > 0: 
            weights[f] = 1 + sig
        else:
            zero_ind[c_zero] += f
            c_zero += 1       
         
    if c_zero > 0:
        zero_ind = zero_ind[0:c_zero]
        weight_zero = (c_zero)/(info['n_ftrs'] + np.sqrt(info['n_ftrs'] - c_zero))
        for z in zero_ind:
            weights[z] = weight_zero
    
    return weights

def calcObjetSignificance(full_pe, part_counts):
    part_tot = part_counts.sum()
    part_e = (part_counts * np.log(part_counts))
    part_tot_e = sum(part_e)
    part_sig = np.zeros(part_counts.shape[0])
    
    part_eRem = (part_tot - part_counts) - (part_tot_e - part_e)/(part_tot - part_counts)
    part_sig[part_eRem > full_pe] = (part_eRem - full_pe)/(full_pe*part_counts) 
    
    return part_sig                            

def peof(features, info):
    peof_factors = np.zeros((info['n_rows'], info['n_ftrs']))
    ftr_weights = calcWeights(features, info)
    ftr_weights_s = [(p, k, new_weights[k]) for p, k in enumerate(sorted(new_weights, key=new_weights.get))]
    
    for p, k, v in ftr_weights_s:
        higher_ftrs = np.array([w[2] for w in ftr_weights_s[p+1:]])
        W_j = higher_ftrs.sum()/higher_ftrs.shape[0]
    
        ftr_counts = np.array(list(features[k].valDict.values()))
        ftr_pe = features[k].calcEntropy(ftr_counts)
        ftr_sig = calcObjetSignificance(ftr_pe, ftr_counts)
     
        higher_counts = np.array(list(itertools.chain.from_iterable([list(feat.valDict.values()) for feat in features[higher_ftrs]])))
        higher_pe = calcPartitionEntropy(higher_counts)
        higher_sig = calcObjetSignificance(higher_pe, higher_counts)

        for i in range(info['n_rows']):
            ftr_val = features[k].vals[i - np.sum(features[k].missing > i)] if i != features[k].missing else np.nan
            higher_vals = [feat.vals[i - np.sum(feat.missing > i)] if i != feat.missing else np.nan for feat in features[higher_ftrs]]
            
            attr_partition = attr_part.index(attr_val)
            attr_cnt = attr_counts[attr_partition]
            attr_sig = attr_partition_sigs[attr_partition]
            
            higher_attrs_partition = higher_attrs_part.index(higher_attrs_val)
            higher_attrs_cnt = higher_attrs_counts[higher_attrs_partition]
            higher_attrs_sig = higher_attrs_partition_sigs[higher_attrs_partition]
            
            peof_factors[i][j] = (w_j * attr_cnt * attr_sig + W_j * higher_attrs_cnt * higher_attrs_sig)/(2*num_rows)
            
        return np.sum(peof_factors, axis =1)
'''
        
        
    