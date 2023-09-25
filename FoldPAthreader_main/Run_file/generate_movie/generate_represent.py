import math
import numpy as np
import os
import sys

out_one_path = sys.argv[1]
out_pdbnum_path = sys.argv[2]
PDB_inter_index = sys.argv[3]
PDB_inter_earlyDM = sys.argv[4]

earlyDM = []
tran2DM = []
allDM = []


with open(out_one_path)as f:
    for line in f:
        if line.split()[1] != 'ini-' and float(line.split()[4]) != 0 and (line.split()[1] == 'exp-' or line.split()[1] == 'ma1-'):
            earlyDM.append(line.split()[3])

        if line.split()[1] != 'ini-' and float(line.split()[4]) == 0 and (line.split()[1] == 'en2-'):
            tran2DM.append(line.split()[3])

        if line.split()[1] != 'ini-' and float(line.split()[4]) == 0:
            allDM.append(line.split()[3])

print("max(earlyDM) = ",max(earlyDM))

my_dict = {}
for i in range(0, 14):
    my_dict[i] = None

cut_tran1 = (float(max(earlyDM))-0.2)/4
print("cut_tran1 = ",cut_tran1)

cut_inter = (float(PDB_inter_earlyDM)-float(max(earlyDM)))/3
print("cut_inter = ",cut_inter)

max_tran2 = float(tran2DM[0])
cut_tran2 = (float(tran2DM[0])-0.5)/3
print("max_tran2 = ",max_tran2)
print("cut_tran2 = ",cut_tran2)

max_all = float(max(allDM))
cut_final = (max_all-max_tran2)/3
print("max_all = ",max_all)
print("cut_final = ",cut_final)

begin_final = 0
num_line = 0
with open(out_one_path)as f:
    for line in f:
        if line.split()[1] != 'ini-':
            if float(line.split()[4]) != 0 and (line.split()[1] == 'exp-' or line.split()[1] == 'ma1-'):
                for i in range(4):
                    if float(line.split()[3]) >= 0.2+cut_tran1*i and float(line.split()[3]) < 0.2+cut_tran1*(i+1):
                        stage2_rep = []
                        stage2_rep.append([line.split()[0],line.split()[2]])
                        my_dict[i] = stage2_rep

            if float(line.split()[4]) != 0 and line.split()[1] != 'exp-' and line.split()[1] != 'ma1-':
                for i in range(3):
                    if float(line.split()[3]) >= float(max(earlyDM))+cut_inter*i and float(line.split()[3]) < float(max(earlyDM))+cut_inter*(i+1):
                        stage2_rep = []
                        stage2_rep.append([line.split()[0],line.split()[2]])
                        my_dict[4+i] = stage2_rep


            if num_line == int(PDB_inter_index):
                stage2_rep = []
                stage2_rep.append([line.split()[0], line.split()[2]])
                my_dict[7] = stage2_rep

            if float(line.split()[4]) == 0:
                if line.split()[1] == 'en2-':
                    begin_final = 1
                if begin_final == 0:
                    for i in range(3):
                        if float(line.split()[3]) >= 0.5+cut_tran2*i and float(line.split()[3]) < 0.5+cut_tran2*(i+1):
                            stage2_rep = []
                            stage2_rep.append([line.split()[0],line.split()[2]])
                            my_dict[7+i] = stage2_rep
                else:
                    for i in range(3):
                        if float(line.split()[3]) >= max_tran2+cut_final*i and float(line.split()[3]) < max_tran2+cut_final*(i+1):
                            stage2_rep = []
                            stage2_rep.append([line.split()[0],line.split()[2]])
                            my_dict[10+i] = stage2_rep
        num_line += 1

with open(out_pdbnum_path, "w+") as outfile:
    outfile.write('initial_state    1'+'\n')
    num = 0
    for key,value in my_dict.items():
        if value is None:
            num += 1
            continue
        else:
            my_dict_array = np.array(my_dict[key])
            min_index = np.argmin(my_dict_array[:, 1])
            if num < 4:
                outfile.write('transition1_state    ' + my_dict[key][min_index][0] + '\n')
            if num >= 4 and num < 6:
                outfile.write('intermediate ' + my_dict[key][min_index][0] + '\n')
            if num >= 6 and num < 7:
                outfile.write('intermediate    ' + str(PDB_inter_index) + '\n')
            if num >= 7 and num < 10:
                outfile.write('transition2_state    ' + my_dict[key][min_index][0] + '\n')
            if num >= 10:
                outfile.write('final_state ' + my_dict[key][min_index][0] + '\n')
            num += 1

