import sys

path = sys.argv[1]
Pro_path = sys.argv[2]

target_list=[]

with open(path+'/top_pro_list')as f:
    for line in f.readlines():
        tar = line.split()[0:]
        target_list.append(tar[0])

cluster_list=[]
center_list=[]
with open(Pro_path)as t:
    for line in t.readlines():
        cluster = line.split()[0:]
        cluster_list.append(cluster)
        center_list.append(cluster[0])

with open(path+'/top_pro_alllist', "w")as outfile:
    for i in range(0, len(target_list)):
        ind = center_list.index(target_list[i])

        for j in range(0, len(cluster_list[ind])):
            outfile.write(cluster_list[ind][j]+"\t"+"\n")
