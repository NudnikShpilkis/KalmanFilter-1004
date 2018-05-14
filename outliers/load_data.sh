#!/bin/bash
/usr/bin/hadoop fs -mkdir outliers

for filename in /scratch/hm74/teaching/big-data/data/GROUP0/*.tsv.gz; do
	FNAME=$(basename "$filename" .tar.gz)
	/usr/bin/hadoop fs -put /scratch/hm74/teaching/big-data/data/GROUP0/"$FNAME"  outliers/"$FNAME"
	echo "$FNAME done"
done

