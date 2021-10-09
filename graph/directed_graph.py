import re
from typing import Tuple, List
from collections import defaultdict
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# 手部关键点
num_nodes = 22
epsilon = 1e-6

# 有向边定义
directed_edges = [(i, j) for i, j in [
    (0,1),(0,2),(2,3),(3,4),(4,5),
    (1,6),(6,7),(7,8),(8,9),
    (1,10),(10,11),(11,12),(12,13),
    (1,14),(14,15),(15,16),(16,17),
    (1,18),(18,19),(19,20),(20,21),(0,0)
]]


def plot_hand_skeleton():
    file_path = "D:\\Programming workspaces\\pyCharm workspace\\hand_gesture_recognition\\DHG2016\\gesture_1\\finger_1\\subject_1\\essai_1\\skeleton_world.txt"
    with open(file_path, 'r') as f:
        line = f.readline()
        l = [float(x) for x in re.split(' ', line)]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    index = 0
    nodes = []
    for i in range(0, num_nodes*3, 3):
        x = l[i]
        y = l[i+1]
        z = l[i+2]
        label = '%d' % (index)
        ax.text(x, y, z, label, color='black')
        ax.scatter(x, y, z, marker='o', color='red')
        index += 1
        nodes.append([x,y,z])
    nodes = np.array(nodes)
    for (i, j) in directed_edges:
        e_x = [nodes[i][0], nodes[j][0]]
        e_y = [nodes[i][1], nodes[j][1]]
        e_z = [nodes[i][2], nodes[j][2]]
        ax.plot(e_x, e_y, e_z, color='red')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def build_digraph_adj_list(edges: List[Tuple]) -> np.ndarray:
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    return graph


def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray) -> np.ndarray:
    # NOTE:
    # 1. The paper assumes that the Incidence matrix is square,
    #    so that the normalized form A @ (D ** -1) is viable.
    #    However, if the incidence matrix is non-square, then
    #    the above normalization won't work.
    #    For now, move the term (D ** -1) to the front
    # 2. It's not too clear whether the degree matrix of the FULL incidence matrix
    #    should be calculated, or just the target/source IMs.
    #    However, target/source IMs are SINGULAR matrices since not all nodes
    #    have incoming/outgoing edges, but the full IM as described by the paper
    #    is also singular, since ±1 is used for target/source nodes.
    #    For now, we'll stick with adding target/source IMs.
    degree_mat = full_im.sum(-1) * np.eye(len(full_im))
    # Since all nodes should have at least some edge, degree matrix is invertible
    inv_degree_mat = np.linalg.inv(degree_mat)
    return (inv_degree_mat @ im) + epsilon

def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    full_graph = source_graph + target_graph
    source_graph = normalize_incidence_matrix(source_graph, full_graph)
    target_graph = normalize_incidence_matrix(target_graph, full_graph)
    return source_graph, target_graph

def build_digraph_adj_matrix(edges: List[Tuple]) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype='float32')
    for edge in edges:
        graph[edge] = 1
    return graph


class Graph:
    def __init__(self):
        super().__init__()
        self.num_nodes = num_nodes
        self.edges = directed_edges
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.edges)


if __name__ == "__main__":
    graph = Graph()
    source_M = graph.source_M
    target_M = graph.target_M
    plt.imshow(source_M, cmap='gray')
    plt.show()
    plt.imshow(target_M, cmap='gray')
    plt.show()
    print(source_M)
    print(target_M)