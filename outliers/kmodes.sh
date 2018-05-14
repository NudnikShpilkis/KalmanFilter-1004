module load python/gnu/3.4.4
module load spark/2.2.0
export PYSPARK_PYTHON='/share/apps/python/3.4.4/bin/python'
export PYSPARK_DRIVER_PYTHON='/share/apps/python/3.4.4/bin/python'

DATA_PATH="outliers/"

SPARKCODE="run.py"

#/usr/bin/hadoop fs -rm -r "test.out"
/usr/bin/hadoop fs -rm -r "$DATA_PATH""$1.out"
spark-submit --conf spark.pyspark.python=/share/apps/python/3.4.4/bin/python "$SPARKCODE" $DATA_PATH$1 $2 $3
/usr/bin/hadoop fs -getmerge "$DATA_PATH$1.out" "results/$1.out"  
