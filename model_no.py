import dgl.function as fn
import dgl
import numpy as np
from kan import KAN
from torch_geometric.nn import global_add_pool,global_mean_pool,SAGPooling,global_max_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from scTransSort import scTransSort
import torch
from torch import nn
import torch.nn.functional as F
from attention_no import Attention_Layern,CrossModalAttention
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func

def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func

# 定义消息传递函数
def message_func(edges):
    # 对边进行消息传递，例如将源节点的特征传递给目标节点
    return {'msg': edges.src['feat']}

# 定义消息聚合函数
def reduce_func(nodes):
    # 聚合邻居节点的消息，例如对邻居节点的特征进行求和
    return {'sum_feat': torch.sum(nodes.mailbox['msg'], dim=1)}





class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)#((57456,128),(2,82062))-->(57456,1)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):
        edge_index = data.edge_index
        edge_u = self.lin_u(data.x)#点的特征
        edge_v = self.lin_v(data.x)#点的特征
        edge_uv = self.lin_edge(data.edge_attr)#边的特征
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr

        out_list = []
        gout_list = []
        for n in range(self.n_iter):#10层卷积层
            out = scatter(out[data.line_graph_edge_index[0]], data.line_graph_edge_index[1], dim_size=edge_attr.size(0),
                          dim=0, reduce='add')
            out = edge_attr + out
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)#求每个点的权重
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))#层权重

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias#s = a * tanh(wg+b)
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)#根据节点的度重复权重的个数
        out = (out_all * scores).sum(-1)
        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)#（27875，128）

        return x


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim, n_iter):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)#多层GNN提取特征，并计算权重

    def forward(self, data):
        data.x = self.mlp(data.x)#表示节点的特征，对节点特征进行归一化和激活
        x = self.line_graph(data)#提取特征（10层GNN）后经过batch_norm和线性层

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    #基于图的注意力传播过程，通过计算节点之间的关系以及边的特征来调整节点之间的信息传递，以增强图神经网络的性能
    def propagate_attention(self, g):

        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))

        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        #g.send_and_recv(eids, custom_src_mul_edge('V_h', 'score'), fn.sum('V_h', 'wV'))
        #g.send_and_recv(eids, message_func('V_h'), reduce_func('V_h'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        return h,e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)

class SRR(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']#70
        num_bond_type = net_params['num_bond_type']#6
        hidden_dim = net_params['hidden_dim']#128
        num_heads = net_params['n_heads']#8
        out_dim = net_params['out_dim']#128
        in_feat_dropout = net_params['in_feat_dropout']#0.0
        dropout = net_params['dropout']#0.0
        n_layers = net_params['L']#3
        self.device = net_params['device']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.lap_pos_enc = net_params['lap_pos_enc']
        in_dim = net_params['num_atom_type']#70
        edge_in_dim = net_params['num_bond_type']#6
        n_iter = net_params['n_iter']#10

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter)

        if self.edge_feat:
            self.embedding_e = nn.Linear(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.graph_pred_linear = nn.Identity()
        self.rmodule = nn.Embedding(86, hidden_dim)#在drugbank中有86中反应类型
        #self.rmodule = nn.Embedding(2, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.lin_sim = nn.Linear(1705, hidden_dim)#处理相似矩阵
        #self.lin_sim = nn.Linear(544, hidden_dim)

    def Fusion(self, sub, sim, data):
        # print("sub"+str(sub.shape))#3251,128
        # print("sim" + str(sim.shape))#128,1706
        #print("data" + str(data.shape))
        Max = global_max_pool(sub, data.batch)
        # print("data.batch"+str(data.batch))
        Mean = global_mean_pool(sub, data.batch)
        # print("max" + str(Max.shape))
        # print("mean" + str(Mean.shape))
        d_g = torch.cat([Max,Mean], dim=-1).type_as(sub)#提取局部和全局特征
        #print("d_g:"+str(d_g.shape))#128,256
        d_g = self.graph_pred_linear(d_g)#恒等映射
        sim = self.lin_sim(sim.float())#线性层（1024，1705）--》（1024，128）
        # print("sim_lin:" + str(sim.shape))#128,128
        global_graph = torch.cat([d_g, sim], dim=-1)
        # print("global_graph:" + str(global_graph.shape))#128,384
        return global_graph
    def forward(self, h_data, t_data, g1, g2, e1, e2, head_finger,tail_finger,rel, sim1, sim2):
    #ead_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl,batch_h_e, batch_t_e,head_finger,tail_finger, rel, sim_h, sim_t
        s_h = self.drug_encoder(h_data)#多个GNN提取特征，包括有计算权重。
        s_t = self.drug_encoder(t_data)#多个GNN提取特征，包括有计算权重。

        h1 = self.in_feat_dropout(s_h)#dropout层
        h2 = self.in_feat_dropout(s_t)#dropout层

        e1 = self.embedding_e(e1.float())#（58440，6）--》（58440，128）#每次都不一样
        e2 = self.embedding_e(e2.float())#(59924,6)-->(59924,128)

        for i,conv in enumerate(self.layers):#3层注意力
            h1,e1 = conv(g1, h1, e1)
            h2,e2 = conv(g2, h2, e2)
        #print("\n")
        # print("h1:"+str(h1.shape))#3282,128
        # print("h2:"+str(h2.shape))#3619, 128
        h = self.Fusion(h1, sim1, h_data)
        t = self.Fusion(h2, sim2, t_data)
        # print("h:"+str(h.shape))#128, 384
        # print("t:"+str(t.shape))#128, 384
        pair = torch.cat([h, t], dim=-1)
        # print("pair:" + str(pair.shape))#2,768
        pair = pair.float()
        rfeat = self.rmodule(rel)
        # print("rfeat:" + str(rfeat.shape))#2,128
        logit = (self.lin(pair) * rfeat).sum(-1)
        return logit
def normalize_and_log(array, target_sum):
    '''
    :param array: 传入的numpy数组
    :param target_sum: 正则化到多少，例如标准为1e4
    :return: 返回对数正则化后的numpy数组
    '''
    # 归一化到目标总和
    normalized_array = array / np.sum(array) * target_sum
    # 使用对数函数
    log_normalized_array = np.log1p(normalized_array)
    return log_normalized_array

class Mymodel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']#70
        num_bond_type = net_params['num_bond_type']#6
        hidden_dim = net_params['hidden_dim']#128
        num_heads = net_params['n_heads']#8
        out_dim = net_params['out_dim']#128
        in_feat_dropout = net_params['in_feat_dropout']#0.0
        dropout = net_params['dropout']#0.0
        n_layers = net_params['L']#3
        self.device = net_params['device']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.lap_pos_enc = net_params['lap_pos_enc']
        in_dim = net_params['num_atom_type']#70
        edge_in_dim = net_params['num_bond_type']#6
        n_iter = net_params['n_iter']#10
        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter)

        if self.edge_feat:
            self.embedding_e = nn.Linear(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        # self.lin = nn.Sequential(
        #     # nn.Linear(hidden_dim * 12, hidden_dim * 6),
        #     # nn.PReLU(),
        #     # nn.Linear(hidden_dim * 6, hidden_dim * 4),
        #     # nn.PReLU(),
        #     nn.Linear(hidden_dim * 4, hidden_dim * 2),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        # )
        self.lin = nn.Sequential(
            KAN([hidden_dim * 4, 128, hidden_dim * 2]),
            KAN([hidden_dim * 2, 256, hidden_dim]),
        )
        # self.lin = nn.Sequential(
        #     nn.Linear(hidden_dim * 12, hidden_dim * 8),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim * 8, hidden_dim * 6),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim * 6, hidden_dim * 4),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim * 4, hidden_dim * 2),
        #     nn.PReLU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        # )
        self.graph_pred_linear = nn.Identity()
        self.rmodule = nn.Embedding(86, hidden_dim)#在drugbank中有86中反应类型
        #self.rmodule = nn.Embedding(2, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.lin_sim = nn.Linear(1705, hidden_dim)#处理相似矩阵
        #self.lin_sim = nn.Linear(544, hidden_dim)
        self.FingerprintNet = FingerprintNet()
        self.scTransSort = scTransSort()
        self.Attention_Layern = Attention_Layern()
    def Fusion(self, sub, data, finger,sim_ht):
        Max = global_max_pool(sub, data.batch)
        Mean = global_mean_pool(sub, data.batch)
        d_g = torch.cat([Max, Mean], dim=-1).type_as(sub)  # 提取局部和全局特征
        d_g = self.graph_pred_linear(d_g)  # 恒等映射
        norm_dg = nn.BatchNorm1d(d_g.shape[1]).to(self.device)
        d_g = norm_dg(d_g)
        # #sim = self.lin_sim(sim.float())#线性层（1024，1706）--》（1024，128）
        # global_graph = torch.cat([d_g, finger,sim_ht], dim=-1)
        #++++++++++++++++++++++++++++++++++++++
        finger_c = torch.cat([finger, finger], dim=-1)
        # sim = self.lin_sim(sim.float())#线性层（1024，1706）--》（1024，128）
        # 使用交叉注意力
        d_g1, finger_c1 = self.Attention_Layern(d_g, finger_c)
        d_f = d_g1 + finger_c1
        norm_df = nn.BatchNorm1d(d_f.shape[1]).to(self.device)
        d_f = norm_df(d_f)
        d_g2, sim_ht1 = self.Attention_Layern(d_f, sim_ht)
        global_feature = d_g2 + sim_ht1
        norm_global = nn.BatchNorm1d(global_feature.shape[1]).to(self.device)
        global_feature = norm_global(global_feature)
        return global_feature
        #+++++++++++++++++++++++=====================
        #global_feature = self.CrossModalAttention(d_g, finger_c, sim_ht)
        # d_g1, f_2 = self.Attention_Layern(d_g, finger_c)
        # d_g2, s_1 = self.Attention_Layern(d_g, sim_ht)
        # f_1, s_2 = self.Attention_Layern(finger_c, sim_ht)
        # dg = d_g1+d_g2
        # f = f_1+f_2
        # s = s_1+s_2
        #
        # dfs = torch.cat([dg,f,s],dim = -1)
        #
        # return dfs

        # return global_graph
    def forward(self, h_data, t_data, g1, g2, e1, e2, head_finger,tail_finger,rel, sim1, sim2,sim_dt_h,sim_dt_t):
        s_h = self.drug_encoder(h_data)#多个GNN提取特征，包括有计算权重。
        s_t = self.drug_encoder(t_data)#多个GNN提取特征，包括有计算权重。

        #处理相似性矩阵
        sim1 = self.lin_sim(sim1.float())
        sim2 = self.lin_sim(sim2.float())
        sim_dt_h = self.lin_sim(sim_dt_h.float())
        sim_dt_t = self.lin_sim(sim_dt_t.float())
        sim_12 = torch.cat([sim1, sim2], dim=-1)
        sim_dt_ht = torch.cat([sim_dt_h, sim_dt_t], dim=-1)

        h1 = self.in_feat_dropout(s_h)#dropout层
        h2 = self.in_feat_dropout(s_t)#dropout层

        e1 = self.embedding_e(e1.float())#（58440，6）--》（58440，128）#每次都不一样
        e2 = self.embedding_e(e2.float())#(59924,6)-->(59924,128)
        finger1,finger2 = self.FingerprintNet(head_finger,tail_finger)
        #finger1,finger2 = self.scTransSort(head_finger,tail_finger)
        for i,conv in enumerate(self.layers):#3层注意力
            h1,e1 = conv(g1, h1, e1)
            h2,e2 = conv(g2, h2, e2)

        #h = self.Fusion(h1, sim1, h_data)
        #t = self.Fusion(h2, sim2, t_data)
        #pair = torch.cat([h, t], dim=-1)
        # print("h1:" + str(h1.shape))#344，128
        # print("h2:" + str(h2.shape))#51，128
        # Max1 = global_max_pool(h1, h_data.batch)
        # Mean1 = global_mean_pool(h1, h_data.batch)
        # Max2 = global_max_pool(h2, t_data.batch)
        # Mean2 = global_mean_pool(h2, t_data.batch)
        # print("max" + str(Max.shape))
        # print("mean" + str(Mean.shape))
        # d_g1 = torch.cat([Max1, Mean1], dim=-1).type_as(h1)  # 提取局部和全局特征
        # d_g2 = torch.cat([Max2, Mean2], dim=-1).type_as(h1)  # 提取局部和全局特征
        # print("d_g1:"+str(d_g1))
        # print("d_g2:"+str(d_g2))
        #print("x:"+str(h1))
        #print("h_data:"+str(h_data))
        d_g1 = self.Fusion(h1,h_data,finger1,sim_12)
        d_g2 = self.Fusion(h2,t_data,finger2,sim_dt_ht)
        pair = torch.cat([d_g1, d_g2], dim=-1)
        pair = pair.float()
        rfeat = self.rmodule(rel)
        # print("pair:" + str(pair.shape))
        # print("rfeat:"+str(rfeat.shape))
        logit = (self.lin(pair) * rfeat).sum(-1)
        # print("logit:"+str(logit))
        # print("logit_shape"+str(logit.shape))
        return logit


def Mymodel_net(net_params):
    return Mymodel(net_params)
def SRR_net(net_params):
    return SRR(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'MyModel': Mymodel_net,
        'SRR':SRR_net
    }

    return models[MODEL_NAME](net_params)

class FingerprintNet(nn.Module):
    def __init__(self,n_filters=4, output_dim=128, dropout=0.5):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.lin = nn.Sequential(
        #     nn.Linear(output_dim  * 3, output_dim * 2),
        #     nn.PReLU(),
        #     nn.Linear(output_dim * 2, output_dim),
        # )
        self.sigmoid = nn.Sigmoid()
        # drug finger
        self.df_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.df_bn1 = nn.BatchNorm1d(n_filters)
        self.df_pool1 = nn.MaxPool1d(3)
        self.df_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.df_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.df_pool2 = nn.MaxPool1d(3)
        self.df_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.df_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.df_pool3 = nn.MaxPool1d(3)
        self.df_fc1 = nn.Linear(464, 512)  # 3712
        self.df_bn4 = nn.BatchNorm1d(512)
        self.df_fc2 = nn.Linear(512, output_dim)  # 2944
        self.df_bn5 = nn.BatchNorm1d(output_dim)
        self.rmodule = nn.Embedding(86,output_dim)

        #融合两个药物
        self.comb_fc1 = nn.Linear(2 * output_dim, 1024)
        self.comb_bn1 = nn.BatchNorm1d(1024)
        self.comb_fc2 = nn.Linear(1024, 128)
        self.comb_bn2 = nn.BatchNorm1d(128)
        self.comb_out = nn.Linear(128, 1)
    def forward(self, head_finger,tail_finger):
        # head_finger = data[4]
        # tail_finger = data[5]
        # rel = data[6]
        # drug finger
        # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # head_finger = self.mlp(head_finger)
        # tail_finger = self.mlp(tail_finger)
        xdf1 = self.df_conv1(head_finger)
        xdf1 = self.df_bn1(xdf1)
        xdf1 = self.relu(xdf1)
        xdf1 = self.df_pool1(xdf1)
        xdf1 = self.df_conv2(xdf1)
        xdf1 = self.df_bn2(xdf1)
        xdf1 = self.relu(xdf1)
        xdf1 = self.df_pool2(xdf1)
        xdf1 = self.df_conv3(xdf1)
        xdf1 = self.df_bn3(xdf1)
        xdf1 = self.relu(xdf1)
        xdf1 = self.df_pool3(xdf1)
        xdf1 = xdf1.view(-1, xdf1.shape[1] * xdf1.shape[2])
        xdf1 = self.df_fc1(xdf1)
        xdf1 = self.df_bn4(xdf1)
        xdf1 = self.relu(xdf1)
        xdf1 = self.dropout(xdf1)
        xdf1 = self.df_fc2(xdf1)
        xdf1 = self.df_bn5(xdf1)

        xdf2 = self.df_conv1(tail_finger)
        xdf2 = self.df_bn1(xdf2)
        xdf2 = self.relu(xdf2)
        xdf2 = self.df_pool1(xdf2)
        xdf2 = self.df_conv2(xdf2)
        xdf2 = self.df_bn2(xdf2)
        xdf2 = self.relu(xdf2)
        xdf2 = self.df_pool2(xdf2)
        xdf2 = self.df_conv3(xdf2)
        xdf2 = self.df_bn3(xdf2)
        xdf2 = self.relu(xdf2)
        xdf2 = self.df_pool3(xdf2)
        xdf2 = xdf2.view(-1, xdf2.shape[1] * xdf2.shape[2])
        xdf2 = self.df_fc1(xdf2)
        xdf2 = self.df_bn4(xdf2)
        xdf2 = self.relu(xdf2)
        xdf2 = self.dropout(xdf2)
        xdf2 = self.df_fc2(xdf2)
        xdf2 = self.df_bn5(xdf2)

        # pair = torch.cat([xdf1,xdf2],dim=1)
        # pair = pair.float()

        # xfusion = self.comb_fc1(pair)
        # xfusion = self.comb_bn1(xfusion)
        # xfusion = self.relu(xfusion)
        # xfusion = self.dropout(xfusion)
        # xfusion = self.comb_fc2(xfusion)
        # xfusion = self.comb_bn2(xfusion)
        #xfusion = self.relu(xfusion)
        #xfusion = self.dropout(xfusion)
        #out = self.comb_out(xfusion)



        # pair = torch.cat([xdf1, xdf2], dim=-1)
        #pair = pair.float()
        # rfeat = self.rmodule(rel)
        # logit = (self.lin(pair) * rfeat).sum(-1)

        # return logit


        # rfeat = self.rmodule(rel)
        #logit = ()
        #return pair.squeeze()
        # return out.squeeze()
        return xdf1,xdf2
