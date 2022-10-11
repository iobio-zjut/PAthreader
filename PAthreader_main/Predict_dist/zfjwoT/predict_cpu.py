from __future__ import print_function

import sys

import numpy as np
import torch
import torch.nn.functional as F
from Network import Network
from esm_predict import msa_trans

a3m_path = sys.argv[1]
feat47_path = sys.argv[2]
msa_64_path = sys.argv[3]
out_path=sys.argv[4]


def main():
    device = torch.device("cpu")
    # Create neural network model (depending on first command line parameter)
    model = Network().eval().to(device)
    # pretrained_dict = torch.load('./model.pt', map_location=lambda storage, loc: storage)
    pretrained_dict = torch.load('./pt-model/model.pt', map_location='cpu')
    model.load_state_dict(pretrained_dict)

    # 序列长度
    with open(a3m_path, 'r') as fa:
        L = fa.readline().strip().__len__()

    # 输入特征
    feat47 = np.memmap(feat47_path + '/' + 'seq.deepmetapsicov.map', dtype=np.float32, mode='r', shape=(1, 47, L, L))
    feat47 = torch.from_numpy(feat47).type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)

    msa_feats, row_att = msa_trans(msa_64_path)
    msa_feats = msa_feats.to(device)
    row_att = row_att.to(device)

    with torch.no_grad():
        output = model(feat47, msa_feats, row_att)
        dist = F.softmax(output, dim=1)

        with open(out_path, "a") as f:
            for wi in range(0, L):
                for wj in range(0, L):
                    probs = dist.data[0, :, wi, wj]
                    f.write("{},{},".format(wi + 1, wj + 1))
                    for dbin in range(37):
                        f.write(" {}".format(probs[dbin]))
                    f.write("\n")
        print("save")


if __name__ == "__main__":
    main()
