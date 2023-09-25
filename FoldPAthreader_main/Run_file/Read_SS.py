import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_pdb', metavar='PATH', default=None,
                    help='initialize first 2 GNN layers with pretrained weights')
parser.add_argument('--output_pt', metavar='PATH', default=None,
                    help='evaluate a trained model')
args = parser.parse_args()

from pyrosetta import *

import numpy as np

np.set_printoptions(threshold=np.inf)
init_cmd = list()
init_cmd.append("-multithreading:interaction_graph_threads 1 -multithreading:total_threads 1")
init_cmd.append("-hb_cen_soft")
init_cmd.append("-detect_disulf -detect_disulf_tolerance 2.0")
init_cmd.append("-relax:dualspace true -relax::minimize_bond_angles -default_max_cycles 200")
init_cmd.append("-mute all")
init_cmd.append("-constant_seed")
init_cmd.append("-read_only_ATOM_entries")
init(" ".join(init_cmd))

def extractSS(pose):
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)
    fp = open(args.output_pt, 'w')
    for ires in range(1, pose.size()+1):
        SS = pose.secstruct(ires)
        print(str(ires)+'\t'+SS, file=fp)

def main():
    init_pose=pose_from_pdb(args.input_pdb)
    extractSS(init_pose)

if __name__== "__main__":
    main()