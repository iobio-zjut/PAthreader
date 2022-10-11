import sys
import argparse
import os
from os.path import isfile, isdir, join
import numpy as np
import torch

def main():

    parser = argparse.ArgumentParser(description="predictor network error")
    parser.add_argument("input",
                        action="store",
                        help="path to input ")
    
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="Specifying # of cpus to use for featurization")
    
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="Reprocessing all feature files")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag ")
    
    args = parser.parse_args()
        
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    
    modelpath = join(base, "pDMscore")


    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import pDMscore as pDM
        

    infilepath = args.input
    infolder = "/".join(infilepath.split("/")[:-1])
    insamplename = infilepath.split("/")[-1][:-4]
    outfolder = "/".join(infilepath.split("/")[:-1])
    feature_file_name = join(outfolder, insamplename+".features.npz")

    if (not isfile(feature_file_name)) or args.reprocess:
        pDM.process((join(infolder, insamplename+".pdb"),
                            feature_file_name,
                            args.verbose))

    if isfile(feature_file_name):

        model = pDM.mypDMscore(twobody_size = 33)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        model.load_state_dict(torch.load(join(modelpath, "best.pkl"), map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()

        with torch.no_grad():
            if args.verbose: print("Predicting for", insamplename)
            (idx, val), (f1d, bert), f2d, dmy = pDM.getData(feature_file_name)
            f1d = torch.Tensor(f1d).to(device)
            f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
            idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
            val = torch.Tensor(val).to(device)

            predict_DMscore = model(idx, val, f1d, f2d)

            f = open('pDMscore', 'a')
            kong = "    "
            np_kong = np.array(kong)
            np_DMscore = np.array(predict_DMscore)
            f.write(str(insamplename) + str(np_kong) + str(np_DMscore)+'\n')


        pDM.clean([insamplename],
                  outfolder,
                  verbose=args.verbose,
                  ensemble=False)
    else:
        print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)
            
            
if __name__== "__main__":
    main()
