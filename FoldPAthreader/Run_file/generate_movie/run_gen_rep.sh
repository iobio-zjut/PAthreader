#!/bin/sh

module load anaconda
source activate pytorch

cd /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie

index=$( sed -n 1'p' $1/fold_simulation/out_inter_sort | awk '{printf $1}' )
echo $index
PDB_inter_index=$( sed -n 1'p' $1/fold_simulation/pdb_inter_index | awk '{printf $1}' )
PDB_inter_earlyDM=$( sed -n $PDB_inter_index'p' $1/fold_simulation/"out_"$index | awk '{printf $4}' )

echo $PDB_inter_index
echo $PDB_inter_earlyDM

python /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie/generate_represent.py $1/fold_simulation/"out_"$index $1/fold_simulation/rep_pdb_index $PDB_inter_index $PDB_inter_earlyDM


mkdir $1/fold_simulation/transition_state $1/fold_simulation/transition_state2
for j in $(seq 1 14)
do
	tran_sta_target=$( sed -n $j'p' $1/fold_simulation/rep_pdb_index | awk '{printf $2}' )
	cp $1/fold_simulation/"output_pdb"$index/"model_"$tran_sta_target".pdb" $1/fold_simulation/transition_state/"state"$j".pdb"
done


cd /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie
for ((i=1; i<=$(ls $1/fold_simulation/transition_state/*.pdb | wc -l); i++))
do
	echo $i
	final_state="$1/seq/ranked_0.pdb"
	file1="$1/fold_simulation/transition_state/state$i.pdb"
    	if [ -e $file1 ]; then
		./TMscore $final_state $file1 > "$1/fold_simulation/tran_files/tmscore_tran"
		
		python /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie/rotate_tran.py $1/fold_simulation "$i"
	fi
done

for j in $(seq 1 14)
do
	state=$( sed -n $j'p' $1/fold_simulation/rep_pdb_index | awk '{printf $1}' )
	mv $1/fold_simulation/transition_state2/"rotate_state"$j".pdb" $1/fold_simulation/transition_state2/$j$state".pdb"
done

conda deactivate