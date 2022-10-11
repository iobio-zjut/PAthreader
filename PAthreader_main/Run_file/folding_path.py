from multiprocessing import Pool
import sys
import os

PATH = sys.argv[1]
path_target = sys.argv[2]

PATH_pdb = PATH + '/PAthreader_database/PDB_AFDB_207187/'
PATH_dist = PATH + '/PAthreader_database/PDB_AFDB_dist/'

def newsin(i):
    os.system(PATH + '/PAthreader/PAthreader -s ' + path_target + '/seq.fasta -c ' + path_target + '/profile_10 -p ' + PATH_pdb +
        target_list[i] + '.pdb -L ' + target_list[i] + ' -D ' + PATH_dist + target_list[i] + '.dist -o ' + path_target + '/stage_foldpath -I 1 -t stage1 -O ' + path_target + '/output/')


target_list = []
with open(path_target + '/stage2_score')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        target_list.append(tar[0])

    p=Pool(10)
    for i in range(50, 500):
        r=p.apply_async(newsin,args=(i,))
    p.close()
    p.join()