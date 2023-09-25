#!/bin/sh


echo "<data_dir>         Path to directory of AlphaFold DB50"
echo "<rosetta_dir>      Path to directory of rosetta"

###=============================================alphafold2=====================================================
pdb_ref="$2/seq/ranked_0.pdb"
if [ ! -s $pdb_ref ]; then
	### modeled by alphafold2
else
	$1/Run_file/read_pdb_fasta $pdb_ref $2/seq.fasta
	echo "pdb_ref already exit ..."
fi

###==============================================foldseek=====================================================
foldseek_file = "$2/foldseek_aln"
if [ ! -s $foldseek_file ]; then
	### run foldseek 
else
	echo "foldseek_file already exit ..."
fi

###============================================early_fold_region=================================================
cd $2
mkdir RMSD
python $1/Run_file/run_TMalign.py $2
python $1/Run_file/ResFscore.py $2
python $1/Run_file/Read_SS.py --input_pdb $2/seq/ranked_0.pdb --output_pt $2/second_structure
python $1/Run_file/intermediate.py $2

###===============================================PAthreader=====================================================
python $1/Run_file/read_pdb_dist.py $2/seq/ranked_0.pdb $2 distance10A 10 $2/rmsd_align_result-fsocre
python $1/Run_file/read_pdb_dist.py $2/seq/ranked_0.pdb $2 distance 20 $2/rmsd_align_result-fsocre
mkdir run_frag_files
cp $1/Run_file/pdb_list_name $2/run_frag_files/
mkdir $2/run_frag_files/temp_pdb_dist $2/run_frag_files/treader_pdb

python $1/Run_file/candidate_structure.py $2

for i in $(seq 1 300)
do
	temp_pdb=$( sed -n $i'p' $2/temp_tmscore_rank_300 | awk '{printf $2}' )
	rename_pdb=$( sed -n $i'p' $2/run_frag_files/pdb_list_name | awk '{printf $2}' )
	cp $data_dir/$temp_pdb $2/run_frag_files/temp_pdb_dist/$rename_pdb".pdb"
	python $1/Run_file/read_pdb_dist.py $2/run_frag_files/temp_pdb_dist/$rename_pdb".pdb" $2/run_frag_files/temp_pdb_dist $rename_pdb".dist" 10 $2/rmsd_align_result-fsocre
done
python $1/Run_file/thread_pdb.py $2

###============================================generate_fragment==================================================
cp $2/seq.fasta $1/Run_file/pdb_list_name $2/run_frag_files/
cd $2/run_frag_files
python $1/Run_file/generate_fragment.py $2/run_frag_files


###============================================folding_simulation==================================================
cd $2
mkdir fold_simulation
cd fold_simulation
mkdir final_pdb output_files output_pdb1 output_pdb2 output_pdb3

cp $2/seq.fasta $2/distance $2/inter_index $2/run_frag_files/top3_ident100.3mers $2/run_frag_files/top3_ident100.6mers $2/fold_simulation
cp $1/Run_file/parameter $2/fold_simulation

decimal=0.01
len=$( sed -n '2p' "seq.fasta" | wc -L )

frag_cycles=$(echo "$decimal * $len" | bc)

threshold=150
threshold2=300
frag_cycles=0

if [ "$len" -lt "$threshold" ]; then
	frag_cycles=1
	echo $frag_cycles
elif [ "$len" -lt "$threshold2" ]; then
	frag_cycles=1.5
	echo $frag_cycles
else
	frag_cycles=2.5
	echo $frag_cycles
fi

$rosetta_dir/main/source/bin/AbinitioRelax.default.linuxgccrelease -in:file:fasta seq.fasta -in:file:frag3 top3_ident100.3mers -in:file:frag9 top3_ident100.6mers -abinitio::increase_cycles $frag_cycles -out:pdb -nstruct 1 -score:set_weights hbond_sr_bb 0.5 rg 0 cenpack 0 ss_pair 0

sort -rn -k2 $2/fold_simulation/out_inter > $2/fold_simulation/out_inter_sort
bash $1/Run_file/generate_movie/run_model_movie.sh $2
bash $1/Run_file/generate_movie/run_gen_rep.sh $2


