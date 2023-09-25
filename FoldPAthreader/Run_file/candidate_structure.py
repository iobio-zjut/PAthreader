import sys
import os

path = sys.argv[1]

os.system("sort -r -k2 " + path + "/temp_tmscore > " + path + "/temp_tmscore_rank")

msta_structure_list = []
num_remove_candidate = 0
with open(path + '/rmsd_align_result-fsocre')as f:
    for line in f.readlines():
        tar = line.split()[1:2]
        msta_structure_list.append(float(tar[0]))
        num_remove_candidate = min(msta_structure_list)

print (int(float(num_remove_candidate)))

os.system("tail -n +" + str(int(float(num_remove_candidate))+1) + " " + path + "/temp_tmscore_rank | head -n 300 > " + path + "/temp_tmscore_rank_300")