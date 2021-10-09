# @Author:Xie Ningwei
# @Date:2021-10-09 15:38:48
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-10-09 15:39:44
import re
import sys
sys.path.append("../")
import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import argparse

# 手部关键点个数
joint_num = 22
# 抽取帧数
frame_num = 25
# 通道数（坐标个数）
channel_num = 3
# 姿态个数，1-6为fine姿态，7-14为coarse姿态
class_num = 14
# 训练：测试比 = 7 : 3
sample_num = 2800
train_sample_num = int(sample_num * 0.7)
test_sample_num = sample_num - train_sample_num


def generate_train_test_data(troncage_file):
    info_list = []
    with open(troncage_file, 'r') as f:
        for line in f:
            l = [int(x) for x in re.split(' ', line)]
            # #gesture - #finger - #subject - #essai - # effective beginning frame - # effective end frame
            info_list.append(l)
    random.shuffle(info_list)
    train_info_list = random.sample(info_list, train_sample_num)
    test_info_list = []
    for info in info_list:
        if info not in train_info_list:
            test_info_list.append(info)

    print("Loading training samples...")
    train_position_matrix = np.zeros((train_sample_num, frame_num, joint_num, channel_num), dtype=np.float)
    train_label = np.zeros((train_sample_num, 1), dtype=np.int)
    for index, info in tqdm(enumerate(train_info_list)):
        sample, label = read_sample(info)
        train_position_matrix[index, :, :, :] = sample
        train_label[index, :] = label

    np.save('..\\data\\train_joint_data.npy', train_position_matrix)
    np.save('..\\data\\train_label.npy', train_label)


    print("\nLoading testing samples...")
    test_position_matrix = np.zeros((test_sample_num, frame_num, joint_num, channel_num), dtype=np.float)
    test_label = np.zeros((test_sample_num, 1), dtype=np.int)
    for index, info in tqdm(enumerate(test_info_list)):
        sample, label = read_sample(info)
        test_position_matrix[index, :, :, :] = sample
        test_label[index, :] = label

    np.save('..\\data\\test_joint_data.npy', test_position_matrix)
    np.save('..\\data\\test_label.npy', test_label)





def read_sample(info):
    position_matrix = []
    sample_path = "..\\DHG2016\\gesture_{}\\finger_{}\\subject_{}\\essai_{}".format(info[0], info[1], info[2], info[3])
    skeleton_file = os.path.join(sample_path, "skeleton_world.txt")
    
    label = info[0] - 1
    with open(skeleton_file, 'r') as f:
        lines = f.readlines()

    start_frame = info[4]
    end_frame = info[5]
    start = None
    end = None
    if (end_frame - start_frame + 1) < frame_num and start_frame == 0:
        start = start_frame
        end = frame_num
    elif (end_frame - start_frame + 1) < frame_num and (end_frame - frame_num + 1) >= 0:
        start = end_frame - frame_num + 1
        end = end_frame + 1
    elif (end_frame - start_frame + 1) >= frame_num:
        start = start_frame
        end = end_frame + 1
    else:
        start = 0
        end = frame_num

    position_matrix = np.zeros((end - start, joint_num, channel_num), dtype=np.float)
    for i, frame_index in enumerate(range(start, end)):
        l = [float(x) for x in re.split(' ', lines[frame_index])]
        for j, coor_index in enumerate(range(0, joint_num * channel_num, channel_num)):
            position_matrix[i, j, 0] = l[coor_index]
            position_matrix[i, j, 1] = l[coor_index + 1]
            position_matrix[i, j, 2] = l[coor_index + 2]

    if position_matrix.shape[0] > frame_num:
        position_matrix = down_sample(position_matrix)

    return position_matrix, label


# 下采样到20帧
def down_sample(position_matrix):
    line = position_matrix.shape[0]
    XperClip = line // frame_num
    p = 1
    k = 0
    clip = np.zeros((frame_num, position_matrix.shape[1], 3))
    for j in range(0, XperClip * (frame_num - 1) + 1, XperClip):
        clip[k,] = position_matrix[j,]
        k = k + 1
    return clip

def parse_args():
    parser = argparse.ArgumentParser(description='Joint Data Generation for Hand Gesture Recognition Based on DGNN')
    parser.add_argument('--troncage_file', default='..\\DHG2016\\informations_troncage_sequences.txt', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generate_train_test_data(args.troncage_file)
    print("Sample number in training set: {}. Sample number in validating set: {}".format(train_sample_num, test_sample_num))