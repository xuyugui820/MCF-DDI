from torch import nn


class Attention_Layer(nn.Module):
    def __init__(self, input_rna=256, input_atac=256):
        super(Attention_Layer, self).__init__()
        self.input_rna = input_rna
        self.input_atac = input_atac
        self.gap = 256
        self.dropout = 0.1
        self.num_classes = 128
        self.n_head = 16
        self.mix_attention_head = self.n_head * 2
        self.attention_dim = self.input_rna

        # 定义交叉注意力层
        self.cross_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head)

        # 具有自注意力的 Transformer 编码器层
        self.encoder_layer_rna = nn.TransformerEncoderLayer(
            d_model=self.gap, dim_feedforward=1024, nhead=self.n_head, dropout=self.dropout
        )

        self.encoder_layer_atac = nn.TransformerEncoderLayer(
            d_model=self.gap, dim_feedforward=1024, nhead=self.n_head, dropout=self.dropout
        )

        # 带有自注意力的位置编码
        # self.positionalEncoding_rna = PositionalEncoding(d_model=self.gap, dropout=dropout)
        # self.positionalEncoding_atac = PositionalEncoding(d_model=self.gap, dropout=dropout)

        # 分类层
        self.pred_layer = nn.Sequential(
            nn.Linear(self.gap, self.gap),
            nn.ReLU(),
            nn.Linear(self.gap, self.num_classes)
        )

    def pretrain(self, x_rna, x_atac):
        x_rna_qkv, x_atac_qkv = x_rna, x_atac
        # 交叉注意力(qkv的三个线性层在nn.MultiheadAttention已经内置了)
        x_rna_att = self.cross_attention_layer(x_rna_qkv, x_atac_qkv, x_atac_qkv)[0]
        x_atac_att = self.cross_attention_layer(x_atac_qkv, x_rna_qkv, x_rna_qkv)[0]

        # 消融实验消去交叉注意力
        # x_rna_att = self.cross_attention_layer(x_rna_qkv, x_rna_qkv, x_rna_qkv)[0]
        # x_atac_att = self.cross_attention_layer(x_atac_qkv, x_atac_qkv, x_atac_qkv)[0]

        # 新的特征
        f_rna = x_rna * 0.5 + x_rna_att * 0.5
        f_atac = x_atac * 0.4 + x_atac_att * 0.6

        # 进行预测的时候需要进行reshape
        # [n_cell, n_feature] -> [n_cell, n_sub_num, n_new_feature=gap]
        f_rna = f_rna.reshape(-1, int(self.input_rna / self.gap), self.gap)
        f_atac = f_atac.reshape(-1, int(self.input_atac / self.gap), self.gap)

        f_rna = f_rna.permute(1, 0, 2)
        f_atac = f_atac.permute(1, 0, 2)
        # 位置编码
        # f_rna = self.positionalEncoding_rna(f_rna)
        # f_atac = self.positionalEncoding_atac(f_atac)
        # Transformer编码器层
        f_rna = self.encoder_layer_rna(f_rna)
        f_atac = self.encoder_layer_atac(f_atac)
        f_rna = f_rna.transpose(0, 1)
        f_atac = f_atac.transpose(0, 1)
        # 平均池化层
        f_rna = f_rna.mean(dim=1)
        f_atac = f_atac.mean(dim=1)
        # 分别用两种模态数据进行预测
        pred_rna = self.pred_layer(f_rna)
        pred_atac = self.pred_layer(f_atac)
        # 返回两种模态数据进行预测的细胞类型结果
        # [n_cell, num_classes]
        return pred_rna, pred_atac

    def extraction_feature(self, x_rna=None, x_atac=None):
        # 为了运行交叉注意力模型，并且为了满足在缺失模态时也能运用预训练模型的参数
        x_1 = x_rna if x_rna is not None else x_atac
        x_2 = x_atac if x_atac is not None else x_rna

        # 交叉注意力
        x_1_att = self.cross_attention_layer(x_1, x_2, x_2)[0]
        x_2_att = self.cross_attention_layer(x_2, x_1, x_1)[0]

        # 新的特征
        f_1 = x_1 * 0.5 + x_1_att * 0.5
        f_2 = x_2 * 0.5 + x_2_att * 0.5

        # f_1 = x_1 * 0.3 + x_1_att * 0.7
        # f_2 = x_2 * 0.3 + x_2_att * 0.7

        # 返回交叉注意力预训练模型提取的双模态数据
        # [n_cell, n_feature]
        return f_1, f_2

    def forward(self, x_rna=None, x_atac=None, ope='extraction_feature'):
        # ope是选择进行预训练(pretrain)还是选择进行特征提取(extraction_feature)
        if ope == 'pretrain':
            return self.pretrain(x_rna, x_atac)
        if ope == 'extraction_feature':
            if x_rna is None:
                return self.extraction_feature(x_rna=None, x_atac=x_atac)
            if x_atac is None:
                return self.extraction_feature(x_rna=x_rna, x_atac=None)
            # 若rna和atac数据均不为空，则提取两者的特征
            return self.extraction_feature(x_rna=x_rna, x_atac=x_atac)


import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self):
        super(CrossModalAttention, self).__init__()
        input_dim=256
        self.query_linear1 = nn.Linear(input_dim, input_dim)
        self.key_linear2 = nn.Linear(input_dim, input_dim)
        self.value_linear2 = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, d_g, finger, sim_ht):
        X1 = d_g
        X2 = finger
        X3 = sim_ht
        Q1 = self.query_linear1(X1)
        K2 = self.key_linear2(X2)
        V2 = self.value_linear2(X2)
        Z12 = self.attention(Q1, K2, V2)

        Q2 = self.query_linear1(X2)
        K3 = self.key_linear2(X3)
        V3 = self.value_linear2(X3)
        Z23 = self.attention(Q2, K3, V3)

        Q3 = self.query_linear1(X3)
        K1 = self.key_linear2(X1)
        V1 = self.value_linear2(X1)
        Z31 = self.attention(Q3, K1, V1)

        Z = torch.cat((Z12, Z23, Z31), dim=-1)
        return Z

    def attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, V)
        return output
