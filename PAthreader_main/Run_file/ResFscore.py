import sys
import os

target_path = sys.argv[1]
length = 0

with open(target_path+"/seq.fasta")as t:
    i = 1
    for line in t.readlines():
        if i == 2:
            length = len(line)
        i += 1
print("length = ",length)

res_align_num=[]
for i in range(length):
    res_align_num.append(0)

list=[]
with open(target_path+"/Template_for_I")as l:
    for line in l.readlines():
            list.append(line.split()[0:1])

for k in range(len(list)):
    t1=" "
    t2=" "
    t3=" "
    res_rmsd=[]

    if os.access(target_path+"/RMSD/"+list[k][0]+".rmsd", os.F_OK):
        with open(target_path+"/RMSD/"+list[k][0]+".rmsd")as f, open(target_path+"/RMSD/"+list[k][0]+".ali")as g:
            for line in f.readlines():
                if line.startswith("RMSD"):
                    break
                res_rmsd.append(line.split()[1:2])

            i = 1
            for line in g.readlines():
                if i == 1:
                    t1 = line
                if i == 2:
                    t2 = line
                if i == 3:
                    t3 = line
                i +=1

    f.close()
    g.close()

    res=0
    align=0

    for i in range(len(t1)-1):
        if t1[i] != "-":
            if t2[i] != " ":
                if float(res_rmsd[align][0]) <=2:
                    res_align_num[res] += 1
                if float(res_rmsd[align][0]) > 2 and float(res_rmsd[align][0]) <= 4:
                    res_align_num[res] += 0.75
                if float(res_rmsd[align][0]) > 4 and float(res_rmsd[align][0]) <= 5:
                    res_align_num[res] += 0.25
                align += 1
            res += 1

with open(target_path+"/ResFsocre", 'w') as outfile:
    for i in range(length):
        outfile.write(str(i) + "\t" + str(res_align_num[i])+"\n")
