# FoldPAthreader
  Protein folding pathway prediction

# Developer
```
 Kailong Zhao
 College of Information Engineering
 Zhejiang University of Technology, Hangzhou 310023, China
 Email: zhaokailong@zjut.edu.cn
```

# Contact
```
 Guijun Zhang, Prof
 College of Information Engineering
 Zhejiang University of Technology, Hangzhou 310023, China
 Email: zgj@zjut.edu.cn
```

# Installation
- Python > 3.5
- PyTorch 1.3
- PyRosetta
- AlphaFold2
- Foldseek

- Download Rosetta3.10 source package from https://www.rosettacommons.org/software/.
(where `$ROSETTA_simulation`=path-to-Rosetta)

- Copy and paste ``"ClassicAbinitio.cc"`` and ``"ClassicAbinitio.hh"`` from ``"code/"`` folder in MMpred package to ``"$ROSETTA_simulation/main/source/src/protocols/abinitio/"`` folder in Rosetta.

- Copy and paste ``"FragmentMover.cc"`` and ``"FragmentMover.hh"`` from ``"code/"`` folder in MMpred package to ``"$ROSETTA_simulation/main/source/src/protocols/simple_moves/"`` folder in Rosetta.

- Compile source code using the following commands:

```
 $> cd $ROSETTA_simulation/main/source/
 $> ./scons.py AbinitioRelax -j<NumOfJobs> mode=release bin
```

- Tested on Ubuntu 20.04 LTS

# Running
```
  FoldPAthreader.sh 

  arguments:
  PATH_file                  path to "FoldPAthreader_main" folder
  PATH_target                path to target protein folder(Place a sequence file named "seq.fasta" in the target protein folder)
```
  
# Example
```
  bash PATH_file/Run_file/FoldPAthreader.sh PATH_file PATH_file/example
```

# Resources
- FoldPAthreader generate multiple structures alignment(MSTA) by searching the AlphaFold DB50, which can be accessed through https://alphafold.ebi.ac.uk and https://foldseek.steineggerlab.workers.dev/afdb50.tar.gz.
  

# DISCLAIMER
  The executable software and the source code of FoldPAthreader is distributed free of charge as it is to any non-commercial users. The authors hold no liabilities to the     performance of the program.
