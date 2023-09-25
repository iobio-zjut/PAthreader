import math
import sys

input = sys.argv[1]
path = sys.argv[2]
dist_name = sys.argv[3]
dist_A = sys.argv[4]
ResFsocre = sys.argv[5]

with open(input)as f, open(ResFsocre)as g, open(path+'/'+dist_name, "w") as outfile:
    vec_index_xyz=[]
    for line in f:
        ATOM = line.split()[0]
        if ATOM == "ATOM":
            CA = line.split()[2]
            if CA=="CA":
                vec_index_xyz.append(line.split()[5:9])


    if float(dist_A) == 10:
        for i in range(len(vec_index_xyz)):
            for j in range(i, len(vec_index_xyz)):
                x1 = float(vec_index_xyz[i][1])
                y1 = float(vec_index_xyz[i][2])
                z1 = float(vec_index_xyz[i][3])
                x2 = float(vec_index_xyz[j][1])
                y2 = float(vec_index_xyz[j][2])
                z2 = float(vec_index_xyz[j][3])
                dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))
                if dist <= float(dist_A) and abs(int(vec_index_xyz[i][0])-int(vec_index_xyz[j][0])) >= 3:
                    outfile.write(str(vec_index_xyz[i][0]) + '\t' + str(vec_index_xyz[j][0]) + '\t' + str(format(dist, '.3f')) + '\t' + '1' + '\n')

    if float(dist_A) == 20:
        vec_Fscore = []
        for line in g:
            vec_Fscore.append(line.split()[2])

        for i in range(len(vec_index_xyz)):
            for j in range(len(vec_index_xyz)):
                x1 = float(vec_index_xyz[i][1])
                y1 = float(vec_index_xyz[i][2])
                z1 = float(vec_index_xyz[i][3])
                x2 = float(vec_index_xyz[j][1])
                y2 = float(vec_index_xyz[j][2])
                z2 = float(vec_index_xyz[j][3])
                dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))
                if dist <= float(dist_A) and abs(int(vec_index_xyz[i][0])-int(vec_index_xyz[j][0])) >= 3:
                    F1 = float(vec_Fscore[int(vec_index_xyz[i][0]) - 1])
                    F2 = float(vec_Fscore[int(vec_index_xyz[j][0]) - 1])
                    if F1 == 0:
                        F1 = 0.01
                    if F2 == 0:
                        F2 = 0.01
                    confidence = 2*F1*F2/(F1+F2)
                    outfile.write(str(vec_index_xyz[i][0]) + ' ' + str(vec_index_xyz[j][0]) + ' ' + str(format(dist, '.3f')) + ' ' + str(format(confidence, '.3f')) + '\n')