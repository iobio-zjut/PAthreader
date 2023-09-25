from multiprocessing import Pool
import sys
import os

path = sys.argv[1]

def newsin(i):
    os.system('/home/data/user/zhanglabs/PAthreader/FoldPAthreader/Run_file/TMalign '+path+'/seq/ranked_0.pdb /home/data/database/AFDB/AFDB-920-lddt09/AFDB_4485529/'+AFDB_list[i]+' > '+path+'/RMSD/'+AFDB_list[i]+'.rmsd')

AFDB_list = []
with open(path + '/foldseek_aln')as f:
    for line in f.readlines():
        AFDB_name = line.split()[1:2]
        AFDB_list.append(AFDB_name[0])

    p=Pool(10)
    for i in range(0, len(AFDB_list)):
        r=p.apply_async(newsin,args=(i,))
    p.close()
    p.join()