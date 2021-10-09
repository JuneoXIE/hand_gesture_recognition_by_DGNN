# @Author:Xie Ningwei
# @Date:2021-10-09 15:47:19
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-10-09 15:47:19
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

pairs = [(j, i) for (i, j) in [
    (0,1),(0,2),(2,3),(3,4),(4,5),
    (1,6),(6,7),(7,8),(8,9),
    (1,10),(10,11),(11,12),(12,13),
    (1,14),(14,15),(15,16),(16,17),
    (1,18),(18,19),(19,20),(20,21),(0,0)
]]

modes = ['train','test']


def generate_bone_data():
    for mode in modes:
        joint_data = np.load('..\\data\\{}_joint_data.npy'.format(mode))
        # (samples (N), # frames (T), # nodes (V), # channels (C))
        N, T, V, C = joint_data.shape

        fp_sp = open_memmap(
            '..\\data\\{}_bone_data.npy'.format(mode),
            dtype='float32',
            mode='w+',
            shape=(N, T, V, C))

        # Copy the joints data to bone placeholder tensor
        fp_sp[:, :, :, :] = joint_data

        for v1, v2 in tqdm(pairs):
            fp_sp[:, :, v1, :] = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]


if __name__ == '__main__':
    generate_bone_data()