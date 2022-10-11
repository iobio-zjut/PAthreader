import numpy as np
import scipy.signal as signal

with open('distance.txt')as f, open('profile_10', "w") as outfile1, open('profile_20', "w") as outfile2:
    for line in f:
        lineDist_comma = []
        res = line.split()[0]
        res1 = res.split(',')[0]
        res2 = res.split(',')[1]

        lineDist=line.split()[1:]
        lineDist = list(map(float, lineDist))

        for i in range(0, 37):
            lineDist_comma.append(lineDist[i:i+1][0])

        x = np.array(lineDist_comma)

        maxinum = x[signal.argrelextrema(x, np.greater)]
        maxinum_index = signal.argrelextrema(x, np.greater)

        lineDist_index = np.argsort(lineDist)

        if len(maxinum) and lineDist_index[36] != 0:
            maxinum_sort_index = np.argsort(maxinum)
            L = len(maxinum_sort_index)
            if L >= 3:
                a = maxinum_index[0][maxinum_sort_index[L - 1]]
                b = maxinum_index[0][maxinum_sort_index[L - 2]]
                c = maxinum_index[0][maxinum_sort_index[L - 3]]
            if L == 2:
                a = maxinum_index[0][maxinum_sort_index[1]]
                b = maxinum_index[0][maxinum_sort_index[0]]
                c = 'NA'
            if L == 1:
                a = maxinum_index[0][maxinum_sort_index[0]]
                b = 'NA'
                c = 'NA'

            if a == 0 or a == 'NA':
                dist_bin1 = "0.00"
            else:
                dist_bin1 = ("%.2f" % (((2 + 0.5 * (a - 1)) + (2 + 0.5 * a)) / 2))
            if b == 0 or b == 'NA':
                dist_bin2 = "0.00"
            else:
                dist_bin2 = ("%.2f" % (((2 + 0.5 * (b - 1)) + (2 + 0.5 * b)) / 2))
            if c == 0 or c == 'NA':
                dist_bin3 = "0.00"
            else:
                dist_bin3 = ("%.2f" % (((2 + 0.5 * (c - 1)) + (2 + 0.5 * c)) / 2))

            if float(dist_bin1) > 0 and float(dist_bin1) <= 10.00 and abs(int(res1)-int(res2)) >= 3 and float(lineDist[a]) > 0.01:
                outfile1.write(str(res1) + '\t' + str(res2) + '\t' + str(dist_bin1) + '\t' + str(format(lineDist[a], '.5f')) + '\t')
                if float(dist_bin2) > 0 and float(dist_bin2) <= 10.00 and abs(int(res1)-int(res2)) >= 3:
                    outfile1.write(str(dist_bin2) + '\t' + str(format(lineDist[b], '.5f')) + '\t')
                else:
                    outfile1.write('0' + '\t' + '0' + '\t')

                if float(dist_bin3) > 0 and float(dist_bin3) <= 10.00 and abs(int(res1)-int(res2)) >= 3:
                    outfile1.write(str(dist_bin3) + '\t' + str(format(lineDist[c], '.5f')))
                else:
                    outfile1.write('0' + '\t' + '0' + '\t')

                outfile1.write("\n")

            if float(dist_bin1) > 0 and float(dist_bin1) <= 20.00 and abs(int(res1)-int(res2)) >= 3 and float(lineDist[a]) > 0.01:
                outfile2.write(str(res1) + '\t' + str(res2) + '\t' + str(dist_bin1) + '\t' + str(format(lineDist[a], '.5f')) + '\t')
                if float(dist_bin2) > 0 and float(dist_bin2) <= 20.00 and abs(int(res1)-int(res2)) >= 3:
                    outfile2.write(str(dist_bin2) + '\t' + str(format(lineDist[b], '.5f')) + '\t')
                else:
                    outfile2.write('0' + '\t' + '0' + '\t')

                if float(dist_bin3) > 0 and float(dist_bin3) <= 20.00 and abs(int(res1)-int(res2)) >= 3:
                    outfile2.write(str(dist_bin3) + '\t' + str(format(lineDist[c], '.5f')))
                else:
                    outfile2.write('0' + '\t' + '0' + '\t')

                outfile2.write("\n")