from multiprocessing import Pool
import sys
import os

PATH = sys.argv[1]
path_target = sys.argv[2]

PATH_pdb = PATH + '/PAthreader_database/PDB_AFDB_207187/'
PATH_dist = PATH + '/PAthreader_database/PDB_AFDB_dist/'

def newsin1(i):
    os.system(PATH + '/PAthreader/PAthreader -s ' + path_target + '/seq.fasta -c ' + path_target + '/profile_10 -p ' + PATH_pdb +
        target_list[i] + '.pdb -L ' + target_list[i] + ' -D ' + PATH_dist + target_list[i] + '.dist -o ' + path_target + '/stage2_score -I 1 -t stage1')

def newsin2(i):
    os.system(PATH + '/PAthreader/PAthreader -s ' + path_target + '/seq.fasta -c ' + path_target + '/profile_20 -p ' + PATH_pdb +
        target_list[i] + '.pdb -L ' + target_list[i] + ' -D ' + PATH_dist + target_list[i] + '.dist2 -o ' + path_target + '/stage2_alignScore -I 10 -t stage2 -O ' + path_target + '/output/')


target_list=[]
with open(path_target+'/R_redun_score')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        target_list.append(tar[0])

    p=Pool(10)
    for i in range(0, len(target_list)):
        r=p.apply_async(newsin1,args=(i,))
    p.close()
    p.join()

os.system("sort -rn -k2 stage2_score > stage2_score_sort")

target_list = []
with open(path_target + '/stage2_score_sort')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        target_list.append(tar[0])

    p=Pool(10)
    for i in range(0, 50):
        r=p.apply_async(newsin2,args=(i,))
    p.close()
    p.join()