#!/bin/bash
module load anaconda
source activate pytorch

cd $1/fold_simulation

index=$( sed -n 1'p' $1/fold_simulation/out_inter_sort | awk '{printf $1}' )
python /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie/find_inter_from_ensemble.py $1/fold_simulation/"out_"$index $1/fold_simulation/pdb_inter_index

# 定义源文件夹和目标文件夹的路径

mkdir $1/fold_simulation/tran_files
mkdir $1/fold_simulation/tran_out_pdb

cp $1/fold_simulation/"output_pdb"$index/model_1.pdb $1/fold_simulation/tran_out_pdb

cd /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie
# 遍历源文件夹中所有的PDB文件
for ((i=1; i<$(ls $1/fold_simulation/"output_pdb"$index/*.pdb | wc -l); i++))
do
	final_state="$1/seq/ranked_0.pdb"
	file1="$1/fold_simulation/"output_pdb"$index/model_$i.pdb"
	file2="$1/fold_simulation/"output_pdb"$index/model_$(($i+1)).pdb"
    	if [ -e $file1 ] && [ -e $file2 ]; then
		echo $i
		./TMscore $final_state $file2 > "$1/fold_simulation/tran_files/tmscore_tran"
		./TMscore $file1 $file2 > "$1/fold_simulation/tran_files/tmscore_qianhou"
		
		python /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie/find.py $1/fold_simulation "$(($i+1))" $index
	fi
done

for ((i=1; i<$(ls $1/fold_simulation/"output_pdb"$index/*.pdb | wc -l); i++))
do
	python /home/data/user/zhanglabs/PAthreader/FoldPAthreader/generate_movie/model_movie.py "$1/fold_simulation/tran_out_pdb" $i $1/fold_simulation
done

conda deactivate 




