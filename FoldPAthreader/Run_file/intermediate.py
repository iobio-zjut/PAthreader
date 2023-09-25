import sys

path = sys.argv[1]
switch = 0
SS = []
with open(path + "/second_structure")as f:
    lines = f.readlines()
    for line in lines:
        SS.append(line.split()[1:][0])

for k in range(len(SS)):
    if k >= 1 and k <= len(SS) - 1:
        if SS[k] == 'B' and SS[k - 1] != 'B' and SS[k + 1] != 'B':
            SS[k] = "L"

Fscore = []
with open(path + "/rmsd_align_result-fsocre")as t:
    for line in t.readlines():
        Fscore.append(line.split()[0:3])

vec_ss = []
all_vec_ss = []
for k in range(len(SS)):
    if k < 1:
        vec_ss.append(Fscore[k][0])
    else:
        if SS[k] == SS[k - 1]:
            vec_ss.append(Fscore[k][0])
        else:
            all_vec_ss.append(vec_ss[0])
            vec_ss.clear()
            vec_ss.append(Fscore[k][0])
    if k == len(SS) - 1:
        all_vec_ss.append(vec_ss[0])
        all_vec_ss.append(len(SS) - 1)

all_vec_ss_01 = []
cut_off = 0.5
success = 0
while success == 0:
    max_Fscore = 0
    all_vec_ss_01.clear()
    inter_len = 0
    small = 0
    for i in range(len(all_vec_ss) - 1):
        for j in range(int(all_vec_ss[i]), int(all_vec_ss[i + 1])):
            if float(Fscore[j][2]) > float(max_Fscore):
                max_Fscore = Fscore[j][2]
        if float(max_Fscore) < float(cut_off):
            all_vec_ss_01.append(0)
        else:
            all_vec_ss_01.append(1)
        max_Fscore = 0

    for i in range(len(all_vec_ss_01)):
        if all_vec_ss_01[i] == 0:
            if (int(all_vec_ss[i + 1])-int(all_vec_ss[i])) <= 4 and i < len(all_vec_ss_01)-1 and i > 0 and all_vec_ss_01[i+1] == 1 and all_vec_ss_01[i-1] == 1:
                all_vec_ss_01[i] = 1
            if (int(all_vec_ss[i + 1])-int(all_vec_ss[i])) <= 4 and i < len(all_vec_ss_01)-1 and i > 0 and all_vec_ss_01[i+1] == 1 and all_vec_ss_01[i-1] == 0:
                all_vec_ss_01[i] = 2
            if (int(all_vec_ss[i + 1])-int(all_vec_ss[i])) <= 4 and i < len(all_vec_ss_01)-1 and i > 0 and all_vec_ss_01[i+1] == 0 and all_vec_ss_01[i-1] == 1:
                all_vec_ss_01[i] = 2
            if (int(all_vec_ss[i + 1])-int(all_vec_ss[i])) <= 4 and i == 0 and all_vec_ss_01[i+1] == 1:
                all_vec_ss_01[i] = 2
            if (int(all_vec_ss[i + 1])-int(all_vec_ss[i])) <= 4 and i == len(all_vec_ss_01)-1 and all_vec_ss_01[i-1] == 1:
                all_vec_ss_01[i] = 2

    for i in range(len(all_vec_ss_01)):
        if all_vec_ss_01[i] == 2:
            small = 1

    for i in range(len(all_vec_ss_01)):
        if all_vec_ss_01[i] == 1 or all_vec_ss_01[i] == 2:
            inter_len += (int(all_vec_ss[i + 1]) - int(all_vec_ss[i]) + 1)
    if float(inter_len) >= float(0.85*len(SS)):
        cut_off += 0.005
    if float(inter_len) < float(0.85*len(SS)) and float(inter_len) >= float(0.75*len(SS)) and small == 1:
        for i in range(len(all_vec_ss_01)):
            if all_vec_ss_01[i] == 2:
                all_vec_ss_01[i] = 0
        success = 1
    if float(inter_len) >= float(0.75 * len(SS)) and small == 0:
        cut_off += 0.005
    if float(inter_len) <= float(0.45 * len(SS)):
        cut_off -= 0.005
    if float(inter_len) < float(0.75 * len(SS)) and float(inter_len) > float(0.45 * len(SS)) :
        success = 1

vec_inter_res = []
with open(path + "/inter_index", 'w') as outfile:
    for i in range(len(all_vec_ss_01)):
        if all_vec_ss_01[i] == 1:
            for g in range(int(all_vec_ss[i]), int(all_vec_ss[i + 1])):
                vec_inter_res.append(int(Fscore[g][0]) + 1)
                outfile.write(str(int(Fscore[g][0]) + 1) + '\n')

with open(path + "/seq/ranked_0.pdb")as t, open(path + "/intermediate.pdb", 'w') as outfile:
    for line_pdb in t.readlines():
        if line_pdb.startswith('ATOM'):
            if int(line_pdb[23:26]) in vec_inter_res:
                outfile.write(str(line_pdb))


