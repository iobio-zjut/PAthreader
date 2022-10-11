# import matplotlib.pyplot as plt
import esm
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import numpy as np

torch.set_grad_enabled(False)

# Data Loading

# This sets up some sequence loading utilities for ESM-1b (`read_sequence`) and the MSA Transformer (`read_msa`).


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()


# ----------------------------------------------------------------------------------
def msa_trans(path_file_a3m):
    msa_data = read_msa(path_file_a3m, 64)

    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    msa_batch_tokens = msa_batch_tokens.cuda()

    msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens)[0].cpu()
    row_attention = msa_transformer.predict_contacts(msa_batch_tokens)[1].cpu()

    return msa_contacts, row_attention
# ----------------------------------------------------------------------------------


# a3m_path = '/home/zfj/data_yt/msa_64_greater'
# files_a3m = os.listdir(a3m_path)
# n = 0
# for file in files_a3m:
#     n += 1
#
#     path_file_a3m = a3m_path + "/" + file  # 获取每一个msa的具体路径
#     msa_data = read_msa(path_file_a3m, 64)
#
#     msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
#     msa_batch_tokens = msa_batch_tokens.cuda()
#
#     msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens)[0].cpu()
#     row_attention = msa_transformer.predict_contacts(msa_batch_tokens)[1].cpu()
#     filename = file.split('.')[0]
#     np.save("/home/zfj/data_yt/Msa_features/" + filename, msa_contacts)
#     np.save("/home/zfj/data_yt/row_att/" + filename, row_attention)
#     print(filename, "保存成功", n)
