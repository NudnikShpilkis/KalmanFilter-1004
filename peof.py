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

    