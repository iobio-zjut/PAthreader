#!/bin/sh

in_fasta=$2"/seq.fasta"
out_folder=$2"/predict_dist"

msa="$out_folder/msa.a3m"

if [ ! -s $msa ]; then
	echo "Round 0: 01  search 'MSA' for the query sequence bu hhblits on databases of UniRef30"
	$1/Predict_dist/msa_search/make_msa.sh $in_fasta $out_folder $msa 10 1000000 $1
else
	echo "Round 0: 01  'MSA' already exit ..."
fi
