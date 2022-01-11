# @Author:Xie Ningwei
# @Date:2021-11-24 10:06:02
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-11-26 15:58:24
import math
import os
import numpy as np
from numpy.lib.format import open_memmap


# 将三维坐标表示的两个向量，计算四元数、旋转矩阵、欧拉角
def VtoRot(v1, v2):
    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)

    if np.linalg.norm(nv1 + nv2) == 0:
        q = [0, [1, 0, 0]]
    else:
        half = (nv1 + nv2) / np.linalg.norm(nv1 + nv2)
        q = [np.dot(nv1,nv2), np.cross(nv1, half)]

    # 四元数，q0为旋转角，（q1,q2,q3）是旋转轴坐标
    q0 = q[0]
    q1 = q[1][0]
    q2 = q[1][1]
    q3 = q[1][2]

    R = np.zeros((3, 3))

    # 根据Rodrigues公式计算旋转矩阵
    R[0, 0] = math.cos(q0) + pow(q1, 2) * (1 - math.cos(q0))
    R[0, 1] = q1 * q2 * (1 - math.cos(q0)) - q3 * math.sin(q0)
    R[0, 2] = q2 * math.sin(q0) + q1 * q3 * (1 - math.cos(q0))
    R[1, 0] = q3 * math.sin(q0) + q1 * q2 * (1 - math.cos(q0))
    R[1, 1] = math.cos(q0) + pow(q2, 2) * (1 - math.cos(q0))
    R[1, 2] = -1 * q1 * math.sin(q0) + q2 * q3 * (1 - math.cos(q0))
    R[2, 0] = -1 * q2 * math.sin(q0) + q1 * q3 * (1 - math.cos(q0))
    R[2, 1] = q1 * math.sin(q0) + q2 * q3 * (1 - math.cos(q0))
    R[2, 2] = math.cos(q0) + pow(q3, 2) * (1 - math.cos(q0))


    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([q0,q1,q2,q3]), R, np.array([x,y,z])


def generate_rotation_data(data_base = os.path.join(os.getcwd(), "data", "train_test_8_2_20fr")):
    modes = ['train','test']
    for mode in modes:
        joint_data = np.load(os.path.join(data_base, '{}_joint_data.npy'.format(mode)))
        # (samples (N), # frames (T), # nodes (V), # channels (C))
        N, T, V, C = joint_data.shape

        # C_rot = 欧拉角 3
        rot_data = open_memmap(
            os.path.join(data_base, '{}_rotation_motion_data.npy'.format(mode)),
            dtype='float32',
            mode='w+',
            shape=(N, T, V, 3))
        rot_data = np.zeros((N, T, V, 3))

        for n in range(N):
            for f in range(T - 1):
                for v in range(V):
                    v1 = joint_data[n, f, v, :]
                    v2 = joint_data[n, f+1, v ,:]
                    if np.linalg.norm(v1) == 0.0 or np.linalg.norm(v1) == 0.0:
                        continue
                    Q, R, E = VtoRot(v1, v2)
                    rot_data[n, f, v, :] = E #np.concatenate((Q, R.flatten(), E), axis=0)
    
    print("Done.")

if __name__ == '__main__':
    generate_rotation_data()
    