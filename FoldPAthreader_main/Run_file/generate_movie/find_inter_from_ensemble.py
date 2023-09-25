import math
import numpy as np
import os
import sys

out_one_path = sys.argv[1]
out_pdbnum_path = sys.argv[2]

en3_inter = 0
en3_inter_ok = 0
num_pdb = 1
with open(out_one_path)as f, open(out_pdbnum_path, "w+") as outfile:
    for line in f:
        if line.split()[1] != 'ini-':
            if float(line.split()[4]) != 0:
                en3_inter = 1
            if float(line.split()[4]) == 0:
                en3_inter_ok = 1
        if en3_inter_ok == 1:
            outfile.write(str(num_pdb-1))
            break
        num_pdb += 1

# with open(out_one_path)as f, open(out_pdbnum_path, "w+") as outfile:
#     for line in f:
#         if line.split()[1] != 'ini-':
#             if float(line.split()[4]) != 0:
#                 en3_inter = 1
#             if float(line.split()[4]) == 0:
#                 en3_inter_ok = 1
#         if en3_inter_ok == 1:
#             outfile.write(str(num_pdb-1))
#             break
#         num_pdb += 1

