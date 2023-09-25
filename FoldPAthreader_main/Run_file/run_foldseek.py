from multiprocessing import Pool
import sys
import os

path = sys.argv[1]

def newsin(i):
    os.system('foldseek easy-search --max-seqs 1000 ' + path +'/seq/ranked_0.pdb ' + AFDB_list[i] + ' ' + foldseek_out_aln[i] + ' ' + foldseek_out_tmp[i])

AFDB_list = []
foldseek_out_aln = []
foldseek_out_tmp = []
with open('/home/data/user/zhanglabs/PAthreader/FoldPAthreader/Run_file/foldseek_datalist')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        AFDB_list.append(tar[0])

        out_aln = line.split()[1:2]
        foldseek_out_aln.append(out_aln[0])

        out_tmp = line.split()[2:3]
        foldseek_out_tmp.append(out_tmp[0])

    p=Pool(10)
    for i in range(0, len(AFDB_list)):
        r=p.apply_async(newsin,args=(i,))
    p.close()
    p.join()