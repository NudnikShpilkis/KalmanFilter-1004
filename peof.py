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

            
        
        
    