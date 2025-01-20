
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
from tqdm import tqdm
from torch_sparse import SparseTensor, fill_diag, sum as sparsesum
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def process_batch(i, nn_idx, center_idx, x, n_points, batch_size, device_index):
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    current_nn_idx = nn_idx[i].reshape(-1).to(torch.long)
    current_center_idx = center_idx[i].reshape(-1).to(torch.long)

    values = torch.ones(current_nn_idx.size(0), dtype=torch.float, device=device)
    adj_matrix = SparseTensor(row=current_center_idx, col=current_nn_idx, value=values,
                              sparse_sizes=(n_points, n_points))

    adj_matrix = adj_matrix.set_value_(None).add(adj_matrix.t()).coalesce()
    adj_matrix = adj_matrix.set_value_(torch.ones(adj_matrix.nnz(), dtype=torch.float, device=device))

    deg = adj_matrix.sum(dim=1).to_dense()
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_matrix * deg_inv_sqrt.view(1, -1)

    adj_matrix = adj_matrix.set_diag(0)
    edge_index = adj_matrix.coo()
    num_edge = edge_index[0].size(0)

    src_nodes, dst_nodes = edge_index[0], edge_index[1]
    shared_src = src_nodes.view(-1, 1).eq(src_nodes.view(1, -1))
    shared_dst = dst_nodes.view(-1, 1).eq(dst_nodes.view(1, -1))
    edge_adj = (shared_src | shared_dst).to(torch.float)
    edge_adj.fill_diagonal_(1.0)

    edge_adj_sparse = SparseTensor(row=edge_adj.nonzero()[:, 0].to(torch.long),
                                   col=edge_adj.nonzero()[:, 1].to(torch.long),
                                   value=edge_adj[edge_adj.nonzero(as_tuple=True)].to(torch.float),
                                   sparse_sizes=edge_adj.shape)

    deg = edge_adj_sparse.sum(dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    eDAD = deg_inv_sqrt.view(-1, 1) * edge_adj_sparse * deg_inv_sqrt.view(1, -1)

    edge_name = [(int(src), int(dst)) for src, dst in zip(edge_index[0], edge_index[1])]

    values = adj_matrix.storage.value()
    indices = torch.stack([adj_matrix.storage.row(), adj_matrix.storage.col()], dim=0).to(torch.long)
    mask = indices[0] != indices[1]
    values = values[mask]
    indices = indices[:, mask]
    adj_matrix = SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=adj_matrix.sizes())
    upper_tri_indices = indices
    num_edge = upper_tri_indices.size(1)
    nedge_name = upper_tri_indices.t().tolist()

    row_index = [item for sublist in nedge_name for item in sublist]
    col_index = list(range(num_edge)) * 2
    data = torch.ones(num_edge * 2, device=device)
    T = SparseTensor(row=torch.tensor(row_index, dtype=torch.long, device=device),
                     col=torch.tensor(col_index, dtype=torch.long, device=device), value=data,
                     sparse_sizes=(adj_matrix.size(0), num_edge))

    x_normalized = torch.nn.functional.normalize(x[i], p=2, dim=1)
    cosine_sim = torch.mm(x_normalized, x_normalized.t())
    distance = 1.0 - cosine_sim
    distance.fill_diagonal_(0)
    edge_distances = distance[upper_tri_indices[0], upper_tri_indices[1]]
    edge_feat = edge_distances.unsqueeze(1)

    directed_adj = torch.zeros_like(distance)
    directed_adj[upper_tri_indices[0], upper_tri_indices[1]] = 1

    direction_feat = torch.zeros((num_edge, 2), dtype=torch.long, device=device)
    direction_feat[directed_adj[upper_tri_indices[0], upper_tri_indices[1]] == 1] = torch.tensor([1, 0],
                                                                                                 dtype=torch.long,
                                                                                                 device=device)
    direction_feat[directed_adj[upper_tri_indices[1], upper_tri_indices[0]] == 1] = torch.tensor([0, 1],
                                                                                                 dtype=torch.long,
                                                                                                 device=device)

    edge_features = torch.cat([edge_feat, direction_feat.to(torch.float)], dim=1)

    return DAD, edge_name, eDAD, T, edge_features


def batch_adjacency_matrices(src, dst, num_points,x ):
    batch_size, _, k = src.size()

    # 构建节点邻接矩阵
    node_adjacency_matrix = torch.zeros((batch_size, num_points, num_points), dtype=torch.float32, device= x.device)
    batch_indices = torch.arange(batch_size).view(batch_size, 1, 1)
    node_adjacency_matrix[batch_indices, src, dst] = 1.0
    node_adjacency_matrix = node_adjacency_matrix + node_adjacency_matrix.transpose(1, 2)
    node_adjacency_matrix = torch.clamp(node_adjacency_matrix, max=1.0)

    # 构建边邻接矩阵（线图邻接矩阵）
    edges = torch.stack((src, dst), dim=-1)  # (batch_size, num_points, k, 2)
    edges = edges.view(batch_size, -1, 2)  # (batch_size, num_edges, 2)
    edge_name = edges
    num_edges = edges.size(1)

    # 初始化边邻接矩阵
    edge_adjacency_matrix = torch.zeros((batch_size, num_edges, num_edges), dtype=torch.float32, device= x.device)

    # 创建边索引张量
    source_edges = edges[:, :, 0]  # (batch_size, num_edges)
    target_edges = edges[:, :, 1]  # (batch_size, num_edges)

    # 扩展为三维张量以便进行比较
    source_edges = source_edges.unsqueeze(1).expand(-1, num_edges, -1)  # (batch_size, num_edges, num_edges)
    target_edges = target_edges.unsqueeze(1).expand(-1, num_edges, -1)  # (batch_size, num_edges, num_edges)

    # 判断边是否相邻，使用向量化操作
    source_eq_source = (source_edges == source_edges.transpose(1, 2))
    source_eq_target = (source_edges == target_edges.transpose(1, 2))
    target_eq_source = (target_edges == source_edges.transpose(1, 2))
    target_eq_target = (target_edges == target_edges.transpose(1, 2))

    is_adjacent = (source_eq_source | source_eq_target | target_eq_source | target_eq_target)

    # 填充边邻接矩阵
    edge_adjacency_matrix[is_adjacent] = 1.0

    # 构建稀疏矩阵T，表示顶点到边的映射
    T = torch.zeros((batch_size,num_edges, num_points), dtype=torch.float32, device= x.device)

    # 从 edges 张量中获取起点和终点索引
    source_node = edges[:, :, 0]  # shape: (batch_size, num_edges)
    target_node = edges[:, :, 1]  # shape: (batch_size, num_edges)

    # 将起点和终点索引扩展到适合 scatter_ 的形状
    # 我们需要在最后添加一个维度，因为 scatter_ 的索引张量需要与 T 的最后一维匹配
    source_nodes = source_node.unsqueeze(-1)  # shape: (batch_size, num_edges, 1)
    target_nodes = target_node.unsqueeze(-1)  # shape: (batch_size, num_edges, 1)

    T.scatter_(2, source_nodes, 1.0)
    T.scatter_(2, target_nodes, 1.0)
    T = torch.clamp(T, max=1.0).transpose(1, 2)

    x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)

    # Compute cosine similarity between all pairs of nodes
    x_normalized = torch.nn.functional.normalize(x, p=2, dim=-1)

    # Transpose the last two dimensions for batched matrix multiplication
    x_normalized_transposed = x_normalized.transpose(-2, -1)  # Swap num_points and feature_dim

    # Compute cosine similarity between all pairs of points within each batch
    cosine_sim = torch.bmm(x_normalized, x_normalized_transposed)
    distance = 1.0 - cosine_sim
    edge_distances = distance[torch.arange(batch_size)[:, None], source_node, target_node]
    edge_feat = edge_distances.unsqueeze(1)

    node_adjacency_matrix = node_adjacency_matrix + torch.eye(num_points, device= x.device).unsqueeze(0).repeat(batch_size, 1, 1)
    edge_adjacency_matrix = edge_adjacency_matrix + torch.eye(num_edges, device= x.device).unsqueeze(0).repeat(batch_size, 1, 1)

    normalized_node_adjacency_matrix = normalize_pytorch(node_adjacency_matrix)
    normalized_edge_adjacency_matrix = normalize_pytorch(edge_adjacency_matrix)

    return normalized_node_adjacency_matrix, normalized_edge_adjacency_matrix,  edge_name,T, edge_feat


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)




def pattern_batch_adjacency_matrices(src, dst, num_points, distance):
    batch_size, num_points, k = src.size()

    # 创建节点和边的邻接矩阵
    node_adjacency_matrices = []
    edge_adjacency_matrices = []
    edge_names = []
    T_matrices = []
    edge_features = []

    for i in range(batch_size):
        # 获取当前批次的src和dst
        src_batch = src[i]
        dst_batch = dst[i]

        # 平展索引并构建节点邻接矩阵
        flat_src = src_batch.flatten()
        flat_dst = dst_batch.flatten()
        values = torch.ones_like(flat_src, dtype=torch.float32, device=src.device)
        indices = torch.stack([flat_src, flat_dst])

        # 创建稀疏张量
        node_adj_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(num_points, num_points),
            dtype=torch.float32,  # 或者使用 values 的 dtype
            device=src.device
        )
        # 对称化、限制值和添加自环
        node_adj_matrix = node_adj_matrix + node_adj_matrix.transpose(0, 1)
        node_adj_matrix = node_adj_matrix.coalesce()
        node_adj_matrix = torch.sparse_coo_tensor(
            indices=node_adj_matrix.indices(),
            values=torch.clamp(node_adj_matrix.values(), max=1.0),
            size=node_adj_matrix.size(),
            dtype=node_adj_matrix.dtype,
            device=node_adj_matrix.device
        )
        eye_sparse = torch.sparse_coo_tensor(
            indices=torch.eye(num_points, dtype=node_adj_matrix.dtype, device=node_adj_matrix.device).nonzero().t(),
            values=torch.ones(num_points, dtype=node_adj_matrix.dtype, device=node_adj_matrix.device),
            size=node_adj_matrix.size(),
            dtype=node_adj_matrix.dtype,
            device=node_adj_matrix.device
        )
        node_adj_matrix = node_adj_matrix + eye_sparse

        # 归一化节点邻接矩阵
        normalized_node_adj_matrix = normalize_pytorch(node_adj_matrix)

        # node_adj_matrix = SparseTensor(row=flat_src, col=flat_dst, value=values,
        #                                sparse_sizes=(num_points, num_points)).to(src.device)
        # node_adj_matrix = node_adj_matrix + node_adj_matrix.t()
        # node_adj_matrix = node_adj_matrix.coalesce()
        # node_adj_matrix = node_adj_matrix.set_value_(torch.clamp(node_adj_matrix.storage.value(), max=1.0))
        #
        # # 添加自环
        # node_adj_matrix = node_adj_matrix + SparseTensor.eye(num_points, device=src.device)
        #
        # # 归一化节点邻接矩阵
        # normalized_node_adj_matrix = normalize_pytorch(node_adj_matrix)
        node_adjacency_matrices.append(normalized_node_adj_matrix)

        # 构建边邻接矩阵（线图邻接矩阵）
        edges = torch.stack((src_batch, dst_batch), dim=-1).view(-1, 2)
        edge_names.append(edges)
        num_edges = edges.size(0)
        source_edges = edges[:, 0].unsqueeze(1).expand(-1, num_edges)
        target_edges = edges[:, 1].unsqueeze(1).expand(-1, num_edges)

        # source_eq_source = (source_edges == source_edges.transpose(0, 1))
        # source_eq_target = (source_edges == target_edges.transpose(0, 1))
        # target_eq_source = (target_edges == source_edges.transpose(0, 1))
        # target_eq_target = (target_edges == target_edges.transpose(0, 1))
        #
        # is_adjacent = (source_eq_source | source_eq_target | target_eq_source | target_eq_target)
        is_adjacent = (
                (source_edges == source_edges.transpose(0, 1)) |
                (source_edges == target_edges.transpose(0, 1)) |
                (target_edges == source_edges.transpose(0, 1)) |
                (target_edges == target_edges.transpose(0, 1))
        )
        nonzero_indices = is_adjacent.nonzero(as_tuple=False).t()
        # nonzero_indices = is_adjacent.nonzero(as_tuple=False)
        values = torch.ones(nonzero_indices.size(1), dtype=torch.float32, device=src.device)

        # edge_adj_matrix = SparseTensor(
        #     row=nonzero_indices[:, 0],
        #     col=nonzero_indices[:, 1],
        #     value=values,
        #     sparse_sizes=(num_edges, num_edges)
        # ).to(src.device)
        # 构造索引张量，shape 为 (2, num_nonzero_elements)
        # indices = nonzero_indices.t()

        # 创建稀疏张量
        edge_adj_matrix = torch.sparse_coo_tensor(
            indices=nonzero_indices,
            values=values,
            size=(num_edges, num_edges),
            dtype=torch.float32,  # 或者使用 values 的 dtype
            device=src.device
        )
        # 添加自环
        # edg_eye_sparse = torch.sparse_coo_tensor(
        #     indices=torch.eye(num_edges, dtype=edge_adj_matrix.dtype, device=edge_adj_matrix.device).nonzero().t(),
        #     values=torch.ones(num_edges, dtype=edge_adj_matrix.dtype, device=edge_adj_matrix.device),
        #     size=edge_adj_matrix.size(),
        #     dtype=edge_adj_matrix.dtype,
        #     device=edge_adj_matrix.device
        # )
        eye_indices = torch.arange(num_edges, device=src.device)
        eye_values = torch.ones(num_edges, dtype=edge_adj_matrix.dtype, device=edge_adj_matrix.device)
        eye_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([eye_indices, eye_indices]),
            values=eye_values,
            size=edge_adj_matrix.size(),
            dtype=edge_adj_matrix.dtype,
            device=edge_adj_matrix.device
        )
        edge_adj_matrix = edge_adj_matrix + eye_sparse

        # # 添加自环
        # edge_adj_matrix = edge_adj_matrix + SparseTensor.eye(num_edges, device=src.device)
        #
        # 归一化边邻接矩阵
        normalized_edge_adj_matrix = normalize_pytorch(edge_adj_matrix)
        edge_adjacency_matrices.append(normalized_edge_adj_matrix)

        # 构建T矩阵
        source_node = edges[:, 0]
        target_node = edges[:, 1]
        edge_indices = torch.arange(num_edges, device=src.device)

        # T_indices = torch.cat([
        #     edge_indices.unsqueeze(1).repeat(1, 2).reshape(-1, 1),
        #     torch.cat([source_node.unsqueeze(1), target_node.unsqueeze(1)], dim=1).reshape(-1, 1)
        # ], dim=1).t()
        #
        # T_values = torch.ones(T_indices.size(1), dtype=torch.float32, device=src.device)
        # T_matrix = SparseTensor(row=T_indices[0], col=T_indices[1], value=T_values,
        #                         sparse_sizes=(num_edges, num_points)).to(src.device)
        # 创建重复的边缘索引和节点对索引
        edge_indices_repeated = edge_indices.unsqueeze(1).repeat(1, 2).view(-1)
        node_pairs = torch.cat([source_node.unsqueeze(1), target_node.unsqueeze(1)], dim=1).view(-1)

        # 构建索引张量
        T_indices = torch.stack([ node_pairs, edge_indices_repeated])

        # 创建值张量
        T_values = torch.ones(T_indices.size(1), dtype=torch.float32, device=src.device)

        # 创建稀疏张量
        T_matrix = torch.sparse_coo_tensor(
            indices=T_indices,
            values=T_values,
            size=( num_points, num_edges),
            dtype=torch.float32,
            device=src.device
        )
        T_matrices.append(T_matrix.coalesce())

        # 计算边特征
        edge_distances = distance[i][flat_src, flat_dst]
        edge_features.append(edge_distances.unsqueeze(1))


    return node_adjacency_matrices, edge_adjacency_matrices, edge_names,  T_matrices, edge_features



def build_matrices(src, dst, num_points,distance,):  #内存消耗过大
    batch_size, num_points, k = src.size()

    # 添加偏移量以确保批次中的索引唯一
    offsets = torch.arange(0, batch_size * num_points, num_points, device=src.device).view(batch_size, 1, 1)
    src_with_offset = src + offsets
    dst_with_offset = dst + offsets

    # 展平所有批次的 src 和 dst
    flat_src = src_with_offset.reshape(-1)
    flat_dst = dst_with_offset.reshape(-1)
    values = torch.ones(flat_src.size(0), dtype=torch.float32, device=src.device)

    # 构建整体的节点邻接矩阵
    node_adj_matrix = SparseTensor(row=flat_src, col=flat_dst, value=values,
                                   sparse_sizes=(batch_size * num_points, batch_size * num_points)).to(src.device)
    node_adj_matrix = node_adj_matrix + node_adj_matrix.t()
    node_adj_matrix = node_adj_matrix.coalesce()
    node_adj_matrix = node_adj_matrix.set_value_(torch.clamp(node_adj_matrix.storage.value(), max=1.0))

    # 添加自环
    eye_indices = torch.arange(0, batch_size * num_points, device=src.device)
    eye_values = torch.ones(batch_size * num_points, dtype=torch.float32, device=src.device)
    eye_matrix = SparseTensor(row=eye_indices, col=eye_indices, value=eye_values,
                              sparse_sizes=(batch_size * num_points, batch_size * num_points)).to(src.device)
    node_adj_matrix = node_adj_matrix + eye_matrix

    # 归一化节点邻接矩阵
    normalized_node_adj_matrix = normalize_pytorch(node_adj_matrix)

    # # 按批次区分节点邻接矩阵
    # node_adjacency_matrices = []
    # for i in range(batch_size):
    #     start_idx = i * num_points
    #     end_idx = (i + 1) * num_points
    #     node_adjacency_matrices.append(normalized_node_adj_matrix[start_idx:end_idx, start_idx:end_idx])

    # 构建整体的边邻接矩阵（线图邻接矩阵）
    edges = torch.stack((flat_src, flat_dst), dim=-1).view(-1, 2)
    num_edges_per_batch = edges.size(0) // batch_size
    num_edges = edges.size(0)
    source_edges = edges[:, 0].unsqueeze(1).expand(-1, num_edges)
    target_edges = edges[:, 1].unsqueeze(1).expand(-1, num_edges)

    source_eq_source = (source_edges == source_edges.transpose(0, 1))
    source_eq_target = (source_edges == target_edges.transpose(0, 1))
    target_eq_source = (target_edges == source_edges.transpose(0, 1))
    target_eq_target = (target_edges == target_edges.transpose(0, 1))

    is_adjacent = (source_eq_source | source_eq_target | target_eq_source | target_eq_target)
    nonzero_indices = is_adjacent.nonzero(as_tuple=False)
    values = torch.ones(nonzero_indices.size(0), dtype=torch.float32, device=src.device)

    edge_adj_matrix = SparseTensor(
        row=nonzero_indices[:, 0],
        col=nonzero_indices[:, 1],
        value=values,
        sparse_sizes=(num_edges, num_edges)
    ).to(src.device)

    # 添加自环
    eye_indices_edges = torch.arange(0, num_edges, device=src.device)
    eye_values_edges = torch.ones(num_edges, dtype=torch.float32, device=src.device)
    eye_matrix_edges = SparseTensor(row=eye_indices_edges, col=eye_indices_edges, value=eye_values_edges,
                                    sparse_sizes=(num_edges, num_edges)).to(src.device)
    edge_adj_matrix = edge_adj_matrix + eye_matrix_edges

    # 归一化边邻接矩阵
    normalized_edge_adj_matrix = normalize_pytorch(edge_adj_matrix)

    # 按批次区分边邻接矩阵
    # edge_adjacency_matrices = []
    # edge_start_indices = torch.arange(0, batch_size * num_edges, num_edges, device=src.device)
    # for start_idx in edge_start_indices:
    #     end_idx = start_idx + num_edges
    #     edge_adjacency_matrices.append(normalized_edge_adj_matrix[start_idx:end_idx, start_idx:end_idx])

    # 构建整体的 T 矩阵
    source_node = edges[:, 0]
    target_node = edges[:, 1]
    edge_indices = torch.arange(num_edges, device=src.device)

    T_indices = torch.cat([
        edge_indices.unsqueeze(1).repeat(1, 2).reshape(-1, 1),
        torch.cat([source_node.unsqueeze(1), target_node.unsqueeze(1)], dim=1).reshape(-1, 1)
    ], dim=1).t()

    T_values = torch.ones(T_indices.size(1), dtype=torch.float32, device=src.device)
    T_matrix = SparseTensor(row=T_indices[0], col=T_indices[1], value=T_values,
                            sparse_sizes=(num_edges, batch_size * num_points)).to(src.device)
    T_matrix = T_matrix.coalesce()

    # # 按批次区分 T 矩阵
    # T_matrices = []
    # for start_idx in edge_start_indices:
    #     end_idx = start_idx + num_points
    #     T_matrices.append(T_matrix[start_idx:end_idx, start_idx:end_idx])

    # 计算边特征
    edge_distances = distance.view(batch_size, num_points, num_points)
    edge_indices = torch.arange(num_edges_per_batch, device=src.device).view(batch_size, num_edges_per_batch)
    flat_src = src_with_offset.view(batch_size, -1)
    flat_dst = dst_with_offset.view(batch_size, -1)

    batch_indices = torch.arange(batch_size, device=src.device).view(-1, 1).repeat(1, num_edges_per_batch).view(-1)
    flat_src_batch = flat_src.view(-1)[batch_indices * num_points + flat_src.view(-1)]
    flat_dst_batch = flat_dst.view(-1)[batch_indices * num_points + flat_dst.view(-1)]

    edge_features = distance[batch_indices, flat_src_batch, flat_dst_batch].unsqueeze(1)
    return normalized_node_adj_matrix, normalized_edge_adj_matrix, edges, T_matrix, edge_features

# 示例调用
def normalize_pytorch(matrix):
    """
       行归一化稀疏矩阵，使得每行的和为 1。
       """
    # 将稀疏矩阵转换为 COO 格式
    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    row, col = indices[0], indices[1]

    # 计算每行的和
    rowsum = torch.bincount(row, weights=values, minlength=matrix.size(0)).to(matrix.device)

    # 计算每行和的倒数
    r_inv = torch.pow(rowsum.float(), -1)
    r_inv[torch.isinf(r_inv)] = 0

    # 归一化值
    normalized_values = values * r_inv[row]

    # 创建归一化的稀疏矩阵
    normalized_matrix = torch.sparse_coo_tensor(
        indices=matrix.indices(),
        values=normalized_values,
        size=matrix.size(),
        dtype=matrix.dtype,
        device=matrix.device
    ).coalesce()

    return normalized_matrix

def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
        adj_matrices, edge_matrices, edge_names, T, Edge_features = [], [], [], [] , []
        # adj_matrices, edge_matrices, edge_names, T, Edge_features = build_matrices(center_idx, nn_idx,n_points, dist)

        adj_matrices, edge_matrices, edge_names, T, Edge_features = pattern_batch_adjacency_matrices(center_idx, nn_idx, n_points, dist)

        # adj_matrices, edge_matrices, edge_names, T, Edge_features =  batch_adjacency_matrices(center_idx, nn_idx, n_points, x)




    return torch.stack((nn_idx, center_idx), dim=0), adj_matrices,edge_matrices, edge_names, T, Edge_features


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index, adj_matrices, edge_matrices, edge_name, TS, Edge_features= dense_knn_matrix(x, self.k * self.dilation, relative_pos)
            # edge_index = edge_index.numpy()


        return self._dilated(edge_index), adj_matrices, edge_matrices, edge_name, TS, Edge_features
