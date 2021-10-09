import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_path, label_path, debug=False, use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.use_mmap = use_mmap
        self.load_data()

    def load_data(self):
        # data: (N, T, V, C)
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.label = np.load(self.label_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)

        # resize data:(N, C, T, V, 1)
        self.data = self.data.transpose((0, 3, 1, 2))
        self.data = np.expand_dims(self.data, axis=-1)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Feeder(Dataset):
    def __init__(self, joint_data_path, bone_data_path, label_path, debug=False, use_mmap=True):
        self.joint_dataset = BaseDataset(joint_data_path, label_path, debug, use_mmap)
        self.bone_dataset = BaseDataset(bone_data_path, label_path, debug, use_mmap)

    def __len__(self):
        return min(len(self.joint_dataset), len(self.bone_dataset))

    def __iter__(self):
        return self

    def __getitem__(self, index):
        joint_data, label, index = self.joint_dataset[index]
        bone_data, label, index = self.bone_dataset[index]

        # Either label is fine
        return joint_data, bone_data, label, index

    def get_mean_map(self):
        """Computes the mean and standard deviation of the dataset"""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def top_k(self, score, top_k):
        # Either dataset can be delegate
        return self.joint_dataset.top_k(score, top_k)


if __name__ == '__main__':
    pass