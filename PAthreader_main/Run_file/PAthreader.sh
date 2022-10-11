#!/bin/sh

echo $2"=-=-=-=-=-=-=-= begin =-=-=-=-=-=-=-="
cd $2
mkdir predict_dist

echo "=={=======> Predict distance =={=======>"
distance="$2/distance.txt"
if [ ! -s $distance ]; then
	cd $1/Predict_dist/msa_search
	./msa_search.sh $1 $2
	
	echo "Round 0: 01  Predict distance"
	cp $1/Predict_dist/delete_greater.py $2/predict_dist
	cd $2/predict_dist
	python delete_greater.py $1

	cd $1/Predict_dist/zfjwoT
	python predict_cpu.py $2/predict_dist/msa2.a3m $2/predict_dist $2/predict_dist/ran64.a3m $2/distance.txt
else
	echo "Round 0: 01  'distance.txt' already exit ..."
fi


echo "=={=======> Thread template =={=======>"
cd $2

python $1/Run_file/Th_txt2dist.py $2

python $1/Run_file/stage_1pathreader.py $1 $2
sort -rn -k2 stage1_score > stage1_score_sort


for j in $(seq 1 500)
do
	top_pro=$( sed -n $j'p' stage1_score_sort | awk '{printf $1}' )
	echo $top_pro >> top_pro_list
done

python $1/Run_file/gen_top_pro_alllist.py $2 $1/PAthreader_database/PDB_AFDB_80_pro_rename

chmod 755 $1/makeblastdb_pdb_afdb_rename/blastp
$1/makeblastdb_pdb_afdb_rename/blastp -query seq.fasta -out blast.fasta -db $1/makeblastdb_pdb_afdb_rename/makeblastdb -outfmt 6 -evalue 1e-5 -num_threads 5

chmod 755 $1/Run_file/Th_Remove_redundant
chmod 755 $1/Run_file/Th_Remove_redundant_30
s=$( sed -n '2p' $2/seq.fasta | wc -L )
if [ "$3" == "true" ];then
	$1/Run_file/Th_Remove_redundant $s
else
	$1/Run_file/Th_Remove_redundant_30 $s
fi

mkdir output
python $1/Run_file/stage_2pathreader.py $1 $2


echo "=={=======> Predict pDMscore =={=======>"
cd $2
python $1/Run_file/predict_DMscore.py $1 $2
$1/Run_file/Align_pDM
echo "Template	rankScore	alignScore	pDMscore" > template_rank
sort -rn -k2 Align_pDM_lineweight >> template_rank


if [ "$4" == "true" ];then
	echo "=={=======> Folding pathway =={=======>"
	python $1/Run_file/folding_path.py $1 $2
	sort -rn -k2 stage_foldpath > stage_foldpath_sort

	cat template_rank stage_foldpath_sort > Template_for_I

	mkdir RMSD

	chmod 755 $1/Run_file/TMalign
	chmod 755 $1/Run_file/TMalign_align
	First_temp=$( sed -n 1'p' Template_for_I | awk '{printf $1}' )
	for i in $(seq 1 500)
	do
		temp=$( sed -n $i'p' Template_for_I | awk '{printf $1}' )
		$1/Run_file/TMalign $2/output/$First_temp".pdb" $2/output/$temp".pdb" > $2/RMSD/$temp".rmsd"
		$1/Run_file/TMalign_align $2/output/$First_temp".pdb" $2/output/$temp".pdb" > $2/RMSD/$temp".ali"
	done

	python $1/Run_file/ResFscore.py $2
	python $1/Run_file/Read_SS.py --input_pdb $2/output/$First_temp".pdb" --output_pt $2/second_structure
	python $1/Run_file/intermediate.py $2
fi

rm -rf RMSD
rm top_pro_list
rm top_pro_alllist
rm Template_for_I
rm stage_foldpath_sort
rm stage_foldpath
rm stage2_score_sort
rm stage2_score
rm stage2_alignScore
rm stage1_score_sort
rm stage1_score
rm R_redun_score
rm pDMscore
rm identity
rm blast.fasta
rm second_structure
rm Align_pDM_lineweight
