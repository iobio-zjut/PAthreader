from multiprocessing import Pool
import sys
import os

path = sys.argv[1]
# PATH_pdb = '/home/data/database/AFDB/AFDB-920-lddt09/AFDB_4485529/'

def newsin(i):
    os.system('/home/data/user/zhanglabs/PAthreader/bin/PAthreader/PAthreader -s ' + path + '/seq.fasta -c ' + path + '/distance10A -p '
              + path + '/run_frag_files/temp_pdb_dist/' + top300_target_list[i] + '.pdb -L ' + top300_target_list[i] + ' -D ' + path +
              '/run_frag_files/temp_pdb_dist/' + top300_target_list[i] + '.dist -o '+ path + '/out_align_score_path -I 1 -t stage1 -O ' + path + '/run_frag_files/treader_pdb/')


top300_target_list = []
with open(path + '/run_frag_files/pdb_list_name')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        top300_target_list.append(tar[0])

    p=Pool(10)
    for i in range(0, 300):
        r=p.apply_async(newsin,args=(i,))
    p.close()
    p.join()