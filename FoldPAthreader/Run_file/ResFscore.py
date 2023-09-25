import sys
import os

path = sys.argv[1]
length = 0

with open(path+"/seq.fasta")as t:
    i = 1
    for line in t.readlines():
        if i == 2:
            length = len(line)
        i += 1

res_align_num=[]
for i in range(length):
    res_align_num.append(0)

num_files = 0
AFDB_rmsd_list=[]
files = os.listdir(path+'/RMSD')
for file in files:
    rm_file = 0
    with open(path + '/RMSD/' + file) as f, open(path + '/temp_tmscore', 'a+') as outfile:
        for line in f.readlines():
            if line.startswith("Chain_1TM-score="):
                outfile.write(file[0:-5] + '\t' + str(line.split()[1:2][0])+'\n')
                if float(line.split()[1:2][0]) < 0.3:
                    rm_file = 1

    f.close()
    if rm_file == 1:
        os.remove(path + '/RMSD/' + file)

files_new = os.listdir(path+'/RMSD')
for file in files_new:
    num_files += 1
    with open(path + '/RMSD/' + file) as f:
        t1 = " "
        t2 = " "
        t3 = " "
        align_i = 0
        stop_read_res_rmsd = 0
        res_rmsd = []
        for line in f.readlines():
            if line.startswith("RMSD"):
                stop_read_res_rmsd = 1
            if line.startswith("align_result:"):
                align_i = 1

            if stop_read_res_rmsd == 0:
                res_rmsd.append(line.split()[1:2])
            if align_i == 2:
                t1 = line
            if align_i == 3:
                t2 = line
            if align_i == 4:
                t3 = line
            if align_i >= 1:
                align_i += 1

        res = 0
        align = 0

        for i in range(len(t1)-1):
            if t1[i] != "-":
                if t3[i] != "-":
                    if float(res_rmsd[align][0]) <= 2:
                        res_align_num[res] += 1
                    if float(res_rmsd[align][0]) > 2 and float(res_rmsd[align][0]) <= 4:
                        res_align_num[res] += 0.75
                    if float(res_rmsd[align][0]) > 4 and float(res_rmsd[align][0]) <= 5:
                        res_align_num[res] += 0.25
                    align += 1
                res += 1

with open(path+"/rmsd_align_result-fsocre", 'w') as outfile:
    for i in range(length):
        outfile.write(str(i) + "\t" + str(res_align_num[i]) + "\t" + str(format(res_align_num[i]/num_files, '.3f')) + "\n")
