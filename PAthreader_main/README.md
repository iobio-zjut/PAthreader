# PAthreader
  Remote homologous template recognition and protein folding pathway prediction

# Developer
            Kailong Zhao
            College of Information Engineering
            Zhejiang University of Technology, Hangzhou 310023, China
            Email: zhaokailong@zjut.edu.cn

# Contact
            Guijun Zhang, Prof
            College of Information Engineering
            Zhejiang University of Technology, Hangzhou 310023, China
            Email: zgj@zjut.edu.cn

# Installation
- Python > 3.5
- PyTorch 1.3
- PyRosetta
- Tested on Ubuntu 20.04 LTS


# Running
```
  PAthreader.sh 

  arguments:
  PATH_file                  path to "PAthreader_main" folder
  PATH_target                path to target protein folder(Place a sequence file named "seq.fasta" in the target protein folder)
  homologou                  redundancy of template (true: Do not remove homologous templates, false: remove homologous templates with a sequence identity ≥ 30%)
  foldpath                   Whether to predict the folding path (true or false)
  (You are advised to use at least 10 cpus)
```
  
# Example
```
  bash PATH_file/Run_file/PAthreader.sh PATH_file PATH_file/example true true
```

# Resources
- The dataset used to run the PAthreader can be accessed through http://zhanglab-bioinf.com/PAthreader/download.html.
- PAthreader generate MSA by searching the UniRef30_2020_03_hhsuite(https://wwwuser.gwdg.de/~compbiol/uniclust/) for template recognition.
  

# DISCLAIMER
  The executable software and the source code of DeepUMQA is distributed free of charge as it is to any non-commercial users. The authors hold no liabilities to the     performance of the program.
