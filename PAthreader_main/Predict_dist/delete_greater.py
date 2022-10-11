import os
import sys
import string
import random

PATH = sys.argv[1]

table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

with open('msa.a3m', 'r') as f_msa:
    lines = f_msa.readlines()
    for line in lines:
        if line.startswith('>'):
            continue
        else:
            with open('msa2.a3m', 'a') as f2:
                line1 = line.translate(table)
                f2.write(line1)

os.system(PATH + '/Predict_dist/alnstats msa2.a3m seq.colstats seq.pairstats > /dev/null')
os.system(PATH + '/Predict_dist/deepmetapsicov_makepredmap \
          seq.colstats \
          seq.pairstats \
          seq.deepmetapsicov.map \
          seq.deepmetapsicov.fix > /dev/null 2>&1')

with open('msa2.a3m', 'r') as f1:
    line_1 = f1.readlines()
    N = len(line_1)
    if N > 64:
        randomList = random.sample(range(1, N), 63)
        for i, line_64 in enumerate(line_1):
            if i == 0:
                with open('ran64.a3m', 'a') as f2:
                    f2.write('>')
                    f2.write('\n')
                    f2.write(line_64)
            elif i in randomList:
                with open('ran64.a3m', 'a') as f2:
                    f2.write('>')
                    f2.write('\n')
                    f2.write(line_64)
    else:
        with open('ran64.a3m', 'a') as f2:
            for line_64 in line_1:
                f2.write('>')
                f2.write('\n')
                f2.write(line_64)