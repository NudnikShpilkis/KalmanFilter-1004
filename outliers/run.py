import sys
import string
from pyspark import SparkContext
from pyspark.conf import SparkConf
from utils import prepRDD
from KModes import KModes

num_args = len(sys.argv) - 1
data_file = sys.argv[1]

if num_args > 1:
    num_clusters = int(sys.argv[2])
else:
    num_clusters = 5

if num_args > 2:
    handle_na = sys.argv[3]
else:
    handle_na = 'None'

sc = SparkContext()
sc.addFile('utils.py')
sc.addFile('KModes.py')

info, prepped_data = prepRDD(sc, data_file, '\t', handle_na)
#prepped_data.saveAsTextFile('test.out')
clusters = KModes(n_clusters = num_clusters)

outlier_preds = sc.parallelize(clusters.fit(sc, prepped_data, info).outliers_)

outlier_preds.saveAsTextFile(str(data_file) + ".out")

sc.stop()
