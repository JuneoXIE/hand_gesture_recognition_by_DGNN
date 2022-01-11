# @Author:Xie Ningwei
# @Date:2021-11-24 12:47:54
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-11-24 12:53:52
import os
import numpy as np
from numpy.lib.format import open_memmap
import argparse
from tqdm import tqdm

modes = {'train', 'test'}
parts = {'joint', 'bone'}


def generate_motion_data(data_base = os.path.join(os.getcwd(), "data", "train_test_7_3_20fr")):
    for mode in modes:
        for part in parts:
            fn = os.path.join(data_base, '{}_{}_data.npy'.format(mode, part))
            if not os.path.exists(fn):
                print('Data does not exist for {}_{} set'.format(mode, part))
                continue

            print('Generating motion data for', mode, part)
            data = np.load(fn)
            (N, T, V, C) = data.shape
            fp_sp = open_memmap(
                os.path.join(data_base, '{}_{}_motion_data.npy'.format(mode, part)),
                dtype='float32',
                mode='w+',
                shape=data.shape)

            # Loop through frames and insert motion difference
            for t in tqdm(range(T - 1)):
                fp_sp[:, t, :, :] = data[:, t + 1, :, :] - data[:, t, :, :]

            # Pad last frame with 0
            fp_sp[:, T - 1, :, :] = 0

if __name__ == '__main__':
    generate_motion_data()