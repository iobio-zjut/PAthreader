import sys

path_pdb = sys.argv[1]
index = sys.argv[2]
path_out = sys.argv[3]

with open(path_pdb+'/model_'+index+'.pdb')as f, open(path_out+'/model_movie.pdb', "a+") as outfile:
    outfile.write('MODEL        '+ index + '\n')
    for line in f:
        ATOM = line.split()[0]
        if ATOM=="ATOM":
            outfile.write(str(line))
        if ATOM=="TER":
            outfile.write('ENDMDL' + '\n')
            break

