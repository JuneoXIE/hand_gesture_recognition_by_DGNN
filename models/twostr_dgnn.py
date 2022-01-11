# @Author:Xie Ningwei
# @Date:2021-11-24 11:13:54
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-11-25 11:22:15
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from graph.directed_graph import Graph


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),   # Conv along the temporal dimension only
            padding=(pad, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BiTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        # NOTE: assuming that temporal convs are shared between node/edge features
        self.tempconv = TemporalConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv(fv), self.tempconv(fe)


class DGNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        # Adaptive block with learnable graphs; shapes (V_node, V_edge)
        self.source_M = nn.Parameter(torch.from_numpy(source_M.astype('float32')))
        self.target_M = nn.Parameter(torch.from_numpy(target_M.astype('float32')))

        # Updating functions
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)

        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fv, fe):
        # `fv` (node features) has shape (N, C, T, V_node)
        # `fe` (edge features) has shape (N, C, T, V_edge)
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape

        # Reshape for matmul, shape: (N, CT, V)
        fv = fv.contiguous().view(N, -1, V_node)
        fe = fe.contiguous().view(N, -1, V_edge)

        # Compute features for node/edge updates
        fe_in_agg = torch.einsum('nce,ev->ncv', fe, self.source_M.transpose(0,1))
        fe_out_agg = torch.einsum('nce,ev->ncv', fe, self.target_M.transpose(0,1))
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)   # Out shape: (N,3,CT,V_nodes)
        fvp = fvp.view(N, 3 * C, T, V_node).contiguous().permute(0,2,3,1)   # (N,T,V_node,3C)
        fvp = self.H_v(fvp).permute(0,3,1,2)    # (N,C_out,T,V_node)
        fvp = self.bn_v(fvp)
        fvp = self.relu(fvp)

        fv_in_agg = torch.einsum('ncv,ve->nce', fv, self.source_M)
        fv_out_agg = torch.einsum('ncv,ve->nce', fv, self.target_M)
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)   # Out shape: (N,3,CT,V_edges)
        fep = fep.view(N, 3 * C, T, V_edge).contiguous().permute(0,2,3,1)   # (N,T,V_edge,3C)
        fep = self.H_e(fep).permute(0,3,1,2)    # (N,C_out,T,V_edge)
        fep = self.bn_e(fep)
        fep = self.relu(fep)
        return fvp, fep


class GraphTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.dgn = DGNBlock(in_channels, out_channels, source_M, target_M)
        self.tcn = BiTemporalConv(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = BiTemporalConv(in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv_res, fe_res = self.residual(fv, fe)
        fv, fe = self.dgn(fv, fe)
        fv, fe = self.tcn(fv, fe)
        fv += fv_res
        fe += fe_res
        return self.relu(fv), self.relu(fe)

class TwoStreamDGNN(nn.Module):
    def __init__(self, num_class=14, num_point=22, num_person=1, graph_args=dict(), in_channels=3):
        super(TwoStreamDGNN, self).__init__()

        self.graph = Graph(**graph_args)

        source_M, target_M = self.graph.source_M, self.graph.target_M
        self.data_bn_v = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_e = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = GraphTemporalConv(3, 32, source_M, target_M, residual=False)
        self.l2 = GraphTemporalConv(32, 64, source_M, target_M)
        self.l3 = GraphTemporalConv(64, 64, source_M, target_M)

        # motion data stream
        self.motion_graph = Graph(**graph_args)
        source_M_motion, target_M_motion = self.motion_graph.source_M, self.motion_graph.target_M
        self.data_bn_v_motion = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_e_motion = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1_motion = GraphTemporalConv(3, 32, source_M_motion, target_M_motion, temp_kernel_size=3, residual=False)
        self.l2_motion = GraphTemporalConv(32, 64, source_M_motion, target_M_motion, temp_kernel_size=5)
        self.l3_motion = GraphTemporalConv(64, 64, source_M_motion, target_M_motion, temp_kernel_size=7, stride=2)

        # concat FC layer
        self.concat_fc = nn.Linear(256, 64)
        self.relu = nn.LeakyReLU(0.5, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, num_class)

        nn.init.kaiming_uniform_(self.concat_fc.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_v, 1)
        bn_init(self.data_bn_e, 1)
        bn_init(self.data_bn_v_motion, 1)
        bn_init(self.data_bn_e_motion, 1)

    def forward(self, fv, fe, fv_motion, fe_motion):
        N, C, T, V_node, M = fv.shape
        _, _, _, V_edge, _ = fe.shape

        # Processing frame data stream
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * C, T)
        fv = self.data_bn_v(fv)
        fv = fv.view(N, M, V_node, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_node)

        fe = fe.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_edge * C, T)
        fe = self.data_bn_e(fe)
        fe = fe.view(N, M, V_edge, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_edge)

        fv, fe = self.l1(fv, fe)
        fv, fe = self.l2(fv, fe)
        fv, fe = self.l3(fv, fe)

            # Shape: (N * M,C,T,V), C is same for fv/fe
        out_channels = fv.size(1)


        # Processing motion data stream
        fv_motion = fv_motion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * C, T)
        fv_motion = self.data_bn_v_motion(fv_motion)
        fv_motion = fv_motion.view(N, M, V_node, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_node)

        fe_motion = fe_motion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_edge * C, T)
        fe_motion = self.data_bn_e_motion(fe_motion)
        fe_motion = fe_motion.view(N, M, V_edge, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_edge)

        fv_motion, fe_motion = self.l1_motion(fv_motion, fe_motion)
        fv_motion, fe_motion = self.l2_motion(fv_motion, fe_motion)
        fv_motion, fe_motion = self.l3_motion(fv_motion, fe_motion)

            # Shape: (N * M,C,T,V), C is same for fv/fe
        out_channels_motion = fv_motion.size(1)
    
        # Performs pooling over both nodes and frames, and over number of persons
        fv = fv.view(N, M, out_channels, -1).mean(3).mean(1)
        fe = fe.view(N, M, out_channels, -1).mean(3).mean(1)

        # Performs pooling over both nodes and frames, and over number of persons
        fv_motion = fv_motion.view(N, M, out_channels_motion, -1).mean(3).mean(1)
        fe_motion = fe_motion.view(N, M, out_channels_motion, -1).mean(3).mean(1)

        # Concat node and edge features
        out = torch.cat((fv, fe), dim=-1)
        out_motion = torch.cat((fv_motion, fe_motion), dim=-1)
        out_two_stream = torch.cat((out, out_motion), dim=-1)

        out_two_stream = self.concat_fc(out_two_stream)
        out_two_stream = self.dropout(out_two_stream)
        out_two_stream = self.fc(out_two_stream)
        out_two_stream = self.relu(out_two_stream)

        return out_two_stream

if __name__ == "__main__":
    
    model = TwoStreamDGNN()
    print(model)

