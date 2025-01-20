
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GCN(nn.Module):

    def __init__(self, nfeat_v, nfeat_e, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, 128, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nfeat_v, 128, nfeat_e, node_layer=True)

        # self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=False)
        # self.gc3 = GraphConvolution(nhid, nfeat_v , nfeat_e, nfeat_e, node_layer=True)


    def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X = F.relu(gc1)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        Z = F.relu(gc2)

        X = self.gc3(X, Z, adj_e, adj_v, T)
        return X

        # return F.log_softmax(X, dim=1)


        # X = F.dropout(X, self.dropout, training=self.training)
        # Z = F.dropout(Z, self.dropout, training=self.training)



        # gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        # X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        # X = F.dropout(X, self.dropout, training=self.training)
        # Z = F.dropout(Z, self.dropout, training=self.training)

        # X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        # # return F.log_softmax(X, dim=1)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            # print("this is a node layer")
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            # print("this is an edge layer")
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):

        batch_size = H_v.shape[0]

        if self.node_layer:
            H_v_list = []
            for i in range(batch_size):
                H_v_i = H_v[i]
                H_e_i = H_e[i]
                adj_v_i = adj_v[i]
                adj_e_i = adj_e[i]
                T_i = T[i]
                multiplier1 = torch.spmm(T_i, torch.diag((H_e_i @ self.p.t()).t()[0])) @ T_i.to_dense().t()
                mask1 = torch.eye(multiplier1.shape[0], device=T_i.device)
                M1 = mask1 * torch.ones(multiplier1.shape[0], device=T_i.device) + (1. - mask1) * multiplier1
                adjusted_A = torch.mul(M1, adj_v_i.to_dense())
                '''
                print("adjusted_A is ", adjusted_A)
                normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
                print("normalized adjusted A is ", normalized_adjusted_A)
                '''
                # to avoid missing feature's influence, we don't normalize the A
                output = torch.mm(adjusted_A, torch.mm(H_v_i, self.weight))
                if self.bias is not None:
                    ret = output + self.bias
                # return ret
                H_v_list.append(ret)
            return torch.stack(H_v_list, dim=0)

        else:
            H_e_list = []
            for i in range(batch_size):
                H_v_i = H_v[i]
                H_e_i = H_e[i]
                adj_v_i = adj_v[i]
                adj_e_i = adj_e[i]
                T_i = T[i]
                multiplier2 = torch.spmm(T_i.t(), torch.diag((H_v_i @ self.p.t()).t()[0])) @ T_i.to_dense()
                mask2 = torch.eye(multiplier2.shape[0], device=T_i.device)
                M3 = mask2 * torch.ones(multiplier2.shape[0],device=T_i.device) + (1. - mask2) * multiplier2
                adjusted_A = torch.mul(M3, adj_e_i.to_dense())
                normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
                output = torch.mm(normalized_adjusted_A, torch.mm(H_e_i, self.weight))
                if self.bias is not None:
                    ret = output + self.bias
                # return H_v, ret
                H_e_list.append(ret)
            return torch.stack(H_e_list, dim=0)

        # if self.node_layer:
        #     multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
        #     mask1 = torch.eye(multiplier1.shape[0])
        #     M1 = mask1 * torch.ones(multiplier1.shape[0]) + (1. - mask1)*multiplier1
        #     adjusted_A = torch.mul(M1, adj_v.to_dense())
        #     '''
        #     print("adjusted_A is ", adjusted_A)
        #     normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
        #     print("normalized adjusted A is ", normalized_adjusted_A)
        #     '''
        #     # to avoid missing feature's influence, we don't normalize the A
        #     output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
        #     if self.bias is not None:
        #         ret = output + self.bias
        #     return ret, H_e
        #
        # else:
        #     multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
        #     mask2 = torch.eye(multiplier2.shape[0])
        #     M3 = mask2 * torch.ones(multiplier2.shape[0]) + (1. - mask2)*multiplier2
        #     adjusted_A = torch.mul(M3, adj_e.to_dense())
        #     normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
        #     output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
        #     if self.bias is not None:
        #         ret = output + self.bias
        #     return H_v, ret

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        # elif conv == 'gc':
        #     self.gconv = GraphConvolution(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class PatternGraphConv2d(nn.Module):
    """
   in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True
    """
    def __init__(self, in_features_v,  in_features_e, nhid, conv='edge', act='relu',  bias=True, norm=None,):
        super(PatternGraphConv2d, self).__init__()
        if conv == 'gc':
            self.gconv = GCN(in_features_v, in_features_e, nhid)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, Z, adj_e, adj_v, T):
        #   def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        return self.gconv(x, Z, adj_e, adj_v, T)

class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index, adj_matrices, edge_matrices, edge_name, T, Edge_features = self.dilated_knn_graph(x, y, relative_pos)

        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class PatternDyGraphConv2d(PatternGraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_features_v,  in_features_e, nhid, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(PatternDyGraphConv2d, self).__init__(in_features_v,  in_features_e, nhid, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index, adj_matrices, edge_matrices, edge_name, T, Edge_features = self.dilated_knn_graph(x, y,relative_pos)

        #     def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        x = x.transpose(2, 1).squeeze(-1)
        x = super(PatternDyGraphConv2d, self).forward(x, Edge_features, edge_matrices, adj_matrices, T)

        return x.reshape(B, -1, H, W).contiguous()

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x

class PatternGrapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    in_features_v, out_features_v, in_features_e, out_features_e,
    """
    def __init__(self, in_channels,in_channels_e, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(PatternGrapher, self).__init__()
        self.channels = in_channels
        self.channels_e = in_channels_e
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = PatternDyGraphConv2d(in_channels, in_channels_e, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels , in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
