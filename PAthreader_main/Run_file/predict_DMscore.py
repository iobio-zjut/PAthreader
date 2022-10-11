from multiprocessing import Pool
import sys
import os

PATH = sys.argv[1]
path_target = sys.argv[2]

def newsin(i):
    os.system('python ' + PATH + '/pDMscore/pDMscore.py ' + path_target + '/output/' + target_list[i] +'.pdb')


target_list = []
with open(path_target + '/stage2_alignScore')as f:
    for line in f.readlines():
        tar = line.split()[0:1]
        target_list.append(tar[0])

    p=Pool(10)
    for i in range(0, len(target_list)):
        r=p.apply_async(newsin,args=(i,))
    p.close()
    p.join()