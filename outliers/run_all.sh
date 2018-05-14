#!/bin/bash
/usr/bin/hadoop fs -mkdir outliers

for filename in /scratch/hm74/teaching/big-data/data/GROUP0/*.tsv.gz; do
	FNAME=$(basename "$filename" .tar.gz)
	./kmodes.sh $FNAME 5 "rep"
	echo "$FNAME done"
done

