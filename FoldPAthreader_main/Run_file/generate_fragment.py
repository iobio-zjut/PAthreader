import multiprocessing
import sys
from pyrosetta import *
import pyrosetta.rosetta.core.pose as pose
import os

init()  # extra_options="-unmute all")

root_path = sys.argv[1]  # 工作目录
# root_path = "/home/data/user/guangxingh/zpx/pyrosetta_template/run_frag_files"
pdb_list = []  # 所有pdb的名字列表
with open(os.path.join(root_path, "pdb_list_name")) as f:
    for line in f.readlines():
        pdb = line.replace('\t', '').replace('\n', '')
        pdb_path = os.path.join(root_path, "treader_pdb", f"{pdb}.pdb")
        if os.path.exists(pdb_path):  # pdb文件存在，才放进列表里
            with open(pdb_path)as ff:
                if "ATOM" in ff.read():
                    pdb_list.append(pdb)

fasta_path = os.path.join(root_path, "seq.fasta")
with open(fasta_path) as f:
    fasta = f.read().split('\n')[1]


def get_info(pdb, pdb_residue_dic):  # 读单个pdb的所有信息，放入pdb_residue_dic字典
    pdb_path = os.path.join(root_path, "treader_pdb", f"{pdb}.pdb")
    pose = pose_from_pdb(pdb_path)
    pdb_info = pose.pdb_info()
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)

    residue_dic = {}
    for index in range(1, 1 + len(pose)):
        temp_info_dic = {}
        real_index = pdb_info.number(index)  # 模板上真实的残基索引
        temp_info_dic['phi'] = pose.phi(index)
        temp_info_dic['psi'] = pose.psi(index)
        temp_info_dic['omega'] = pose.omega(index)
        temp_info_dic['secstruct'] = pose.secstruct(index)
        residue_dic[real_index] = temp_info_dic

    pdb_residue_dic[pdb] = residue_dic


if __name__ == '__main__':
    pool = multiprocessing.Pool(10)
    pdb_residue_dic = multiprocessing.Manager().dict()

    for pdb in pdb_list:
        pool.apply_async(get_info, (pdb, pdb_residue_dic,))

    pool.close()
    pool.join()

    with open(os.path.join(root_path, "top3_ident100.3mers"), 'w') as f:
        iterator = iter(range(1, 1 + len(fasta) - 2))
        for index in iterator:
            f.write(f"position:{index:>14} neighbors:            3")
            f.write('\n')
            f.write('\n')
            counter = 200
            while (counter > 0):
                for pdb in pdb_list:
                    residue_dic = pdb_residue_dic[pdb]
                    if index in residue_dic.keys() and index + 1 in residue_dic.keys() and index + 2 in residue_dic.keys():
                        for sub_index in range(index, index + 3):
                            temp_info_dic = residue_dic[sub_index]
                            pdb_name = pdb
                            secstruct = temp_info_dic['secstruct']
                            phi = temp_info_dic['phi']
                            psi = temp_info_dic['psi']
                            omega = temp_info_dic['omega']
                            f.write(f" {pdb_name}         {secstruct}{phi:>9.3f}{psi:>9.3f}{omega:>9.3f}"+'\n')
                        f.write('\n')
                        counter = counter - 1
                        if counter <= 0:
                            break

    with open(os.path.join(root_path, "top3_ident100.6mers"), 'w') as f:
        iterator = iter(range(1, 1 + len(fasta) - 5))
        for index in iterator:
            f.write(f"position:{index:>14} neighbors:            6")
            f.write('\n')
            f.write('\n')
            counter = 200
            while (counter > 0):
                for pdb in pdb_list:
                    residue_dic = pdb_residue_dic[pdb]
                    if index in residue_dic.keys() and index + 1 in residue_dic.keys() and index + 2 in residue_dic.keys() \
                            and index + 3 in residue_dic.keys() and index + 4 in residue_dic.keys() and index + 5 in residue_dic.keys():
                        for sub_index in range(index, index + 6):
                            temp_info_dic = residue_dic[sub_index]
                            pdb_name = pdb
                            secstruct = temp_info_dic['secstruct']
                            phi = temp_info_dic['phi']
                            psi = temp_info_dic['psi']
                            omega = temp_info_dic['omega']
                            f.write(f" {pdb_name}         {secstruct}{phi:>9.3f}{psi:>9.3f}{omega:>9.3f}"+'\n')
                        f.write('\n')
                        counter = counter - 1
                        if counter <= 0:
                            break
