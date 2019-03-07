#!/bin/bash
#SBATCH -N 1
#SBATCH -J Emb_Link
#SBATCH -t 2-00:00:00
#SBATCH -p batch

cd $HOME/Eigenmaps/

declare -a arr_dataset=("blogcatalog" "cornell" "ppi" "pubmed" "texas" "washington" "wisconsin" "co-author" "flickr" "microsoft" "p2p-gnutella31" "wikipedia" "wikivote")
dim=64

for dataset in "${arr_dataset[@]}"
do
	for i in $(seq 0 4);
	do
		mycommand="python -W ignore gem_eigenmaps_link.py --dataset $dataset\
											 --dimension $dim --fold $i"
		echo $mycommand
		$mycommand
	done
done