from multiprocessing import Pool
import sys
import os

PATH = sys.argv[1]
path_target = sys.argv[2]

PATH_pdb = PATH + '/PAthreader_database/PDB_AFDB_207187/'
PATH_dist = PATH + '/PAthreader_database/PAcluster80_pro/'

def newsin(i):
    os.system(PATH + '/PAthreader/PAthreader -s ' + path_target + '/seq.fasta -c ' + path_target + '/profile_10 -p ' + PATH_pdb +
        target_list[i] + '.pdb -L ' + target_list[i] + ' -D ' + PATH_dist + target_list[i] + '.pro -o ' + path_target + '/stage1_score -I 1 -t stage1')

target_list = []
with open(PATH + '/PAthreader_database/list56804_rename')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        target_list.append(tar[0])

    p=Pool(10)
    for i in range(0, len(target_list)):
        p.apply_async(newsin,args=(i,))
    p.close()
    p.join()