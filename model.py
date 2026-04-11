import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class LSTMModel(nn.Module):
    def __init__(self, nx,ny,hidden_size,num_layer,pred_len, drop_rate):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size=hidden_size
        self.pred_len = pred_len
        self.drop = nn.Dropout(drop_rate)
        self.lstm = nn.LSTM(self.nx, self.hidden_size,num_layers=num_layer, batch_first=True, bidirectional=False)
        self.dense = nn.Linear(self.hidden_size, self.pred_len*self.ny)
    def forward(self, x):
        B, N, T, _ = x.shape
        x_in = x.reshape(B * N, T, -1)
        lstm_out,_ = self.lstm(x_in)
        mlp_out = self.dense(lstm_out[:,-1,:])

        return mlp_out.reshape(B, N, self.pred_len, self.ny)


class STGNNModel(nn.Module):
    def __init__(self, nx, ny, num_nodes, edge_index, hidden_size, num_layer, pred_len, drop_rate, device):
        super(STGNNModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.drop = nn.Dropout(drop_rate)

        # 将稀疏的 edge_index 转换为稠密矩阵 [N, N]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)

        # 增加自环 (保证节点保留自身信息)
        adj = adj + torch.eye(num_nodes, device=device)
        adj = (adj > 0).float()

        # 传统 GCN 的对称归一化: D^{-1} A
        deg = adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_adj = deg_inv.view(-1, 1) * adj

        self.register_buffer('norm_adj', norm_adj)

        # ==========================================
        # 2. 网络层定义
        # ==========================================
        # GCN 权重层
        self.W_gcn1 = nn.Linear(nx, hidden_size)
        # LSTM 时序层
        self.lstm = nn.LSTM(nx, hidden_size, num_layers=num_layer, batch_first=True)
        self.lstm_g = nn.LSTM(hidden_size, hidden_size, num_layers=num_layer, batch_first=True)
        # 多步预测全连接层 (直接输出 pred_len * ny)
        self.dense = nn.Linear(self.hidden_size*2, self.pred_len*self.ny)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, nF = x.shape

        # 将时间维度 T 提到前面，方便并行对每个时间步做图卷积 -> [B, T, N, F]
        x_trans = x.permute(0, 2, 1, 3)

        # --- 阶段一: 空间特征提取 (GCN) ---
        h1 = self.W_gcn1(x_trans)
        # 使用 einsum 快速完成矩阵乘法: norm_adj[N, N] * h1[B, T, N, H] -> [B, T, N, H]
        h1 = torch.einsum('ij,btjf->btif', self.norm_adj, h1)
        h1 = F.gelu(h1)

        # 还原维度为 [B, N, T, H]
        gcn_out = h1.permute(0, 2, 1, 3)
        h_gcn = gcn_out.reshape(B*N, T,-1)
        h_out,_ = self.lstm_g(h_gcn)
        ST_out = h_out.reshape(B, N, T, -1)
        # --- 阶段二: 时间序列建模 (LSTM) ---
        # 拉平进行独立站点 LSTM 推演: [B*N, T, H]
        lstm_in = x.reshape(B * N, T, -1)
        last_state, _ = self.lstm(lstm_in)

        # 截取最后一个时间步的状态: [B*N, H]
        lstm_out = last_state.reshape(B,N,T,-1)
        h = torch.cat((lstm_out, ST_out), dim=-1)  # h  ---> [B,N,T,H*2]
        # --- 阶段三: 多步解码 ---
        out = self.dense(h[:, :, -1, :])  # [B,N, pred_len * ny]
        # 还原为您需要的输出维度
        return out.reshape(B, N, self.pred_len, self.ny)

# ==========================================
# 核心层：物理启发的滞后图卷积 (Physics-Guided GCN)
# ==========================================
class PhysicsGuidedGCN(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(PhysicsGuidedGCN, self).__init__()

        self.W = nn.Linear(in_features, hidden_size)
        self.lag_weights = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1)
        )

    def forward(self, x,A_list):
        """
        x: [Batch, N, T, F]
        """
        B, N, T, F_in = x.shape
        max_lag = A_list.shape[1]

        # 调整维度以适应 einsum: [B, T, N, F]
        x_trans = x.permute(0, 2, 1, 3)
        A_list = A_list.to(dtype=x_trans.dtype, device=x_trans.device)
        # 初始化输出容器 [B, T, N, F]
        out_agg = torch.zeros(B, T, N, F_in, device=x.device)

        for lag in range(max_lag):
            norm_A_k = A_list[:,lag,:]
            # ==========================================
            # 核心1：构造多重滞后矩阵，实现上游t-1时刻的水流到下游t时刻
            if lag == 0:
                x_lagged = x_trans
            else:
                x_lagged = torch.roll(x_trans, shifts=lag, dims=1)
                x_lagged[:, :lag, :, :] = 0.0
            # ==========================================
            lag_tensor = torch.tensor([[float(lag)]], dtype=x_trans.dtype, device=x_trans.device)
            dynamic_weight = self.lag_weights(lag_tensor)
            # 核心2：实现上游节点的水流汇到下游
            agg = torch.einsum('bij,btjf->btif', norm_A_k, x_lagged)
            out_agg += agg * dynamic_weight
            # ==========================================
        # [B, N, T, F]
        out_final = out_agg.permute(0, 2, 1, 3)

        return F.gelu(self.W(out_final))

class TemporalModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        # x: [B,N,T,F]
        B, N, T, F = x.shape
        x_in = x.reshape(B * N, T, F)
        out, _ = self.lstm(x_in)
        return out.reshape(B, N, T, -1)

class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(GRN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.skip_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.skip_proj(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)


class TFTDecoderHead(nn.Module):
    def __init__(self, hidden_size, pred_len, out_dim, num_heads=4, dropout=0.1):
        super(TFTDecoderHead, self).__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim

        # 1. 历史序列的特征提纯
        self.historical_grn = GRN(hidden_size, hidden_size, dropout)

        # 2. 多头自注意力机制 (提取全局时序依赖)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        # 3. 预测步长的位置编码 (生成未来 pred_len 步的 Query)
        # 这里用一组可学习的参数来代表未来时间的 Query
        self.future_queries = nn.Parameter(torch.randn(1, 1, pred_len, hidden_size))
        # 4. 最终输出的 GRN
        self.output_grn = GRN(hidden_size, hidden_size, dropout)
        # 5. 映射到预测值
        self.final_proj = nn.Linear(hidden_size, out_dim)

    def forward(self, st_features):
        # st_features: [B, N, T, H] (来自 GNN+LSTM 的输出)
        B, N, T, H = st_features.shape
        # 将 N 和 B 合并以适配 Attention 输入 [B*N, T, H]
        x = st_features.reshape(B * N, T, H)
        # 经过 GRN 提纯
        v = self.historical_grn(x)  # Value & Key
        # 扩展 future queries 以匹配 Batch * Nodes
        q = self.future_queries.expand(B * N, -1, -1, -1).reshape(B * N, self.pred_len, H)
        # Multi-Head Attention: Q=未来预测步, K=V=历史序列
        # 注意力机制会决定未来每一步分别需要重点关注历史的哪些时刻
        attn_out, attn_weights = self.attention(q, v, v)
        # 残差连接与归一化
        attn_out = self.attn_layer_norm(attn_out + q)
        # 通过输出 GRN
        out = self.output_grn(attn_out)
        # 映射到最终维度 [B*N, pred_len, out_dim]
        out = self.final_proj(out)
        return out.reshape(B, N, self.pred_len, self.out_dim)



class PhysicsSTGNN(nn.Module):
    def __init__(self, nx,ny,  hidden_size,num_layer,pred_len, drop_rate):
        super(PhysicsSTGNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.pred_len = pred_len
        self.lstm = TemporalModule(self.nx , self.hidden_size,num_layers=num_layer)
        self.gnn = PhysicsGuidedGCN(self.nx, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size*2, self.hidden_size)
        # 接入 TFT 解码头 (替换掉原来的 MLP)
        self.tft_decoder = TFTDecoderHead(
            hidden_size=hidden_size,
            pred_len=pred_len,
            out_dim=ny,
            dropout=drop_rate
        )
    def forward(self, x,A_list):
        # x: [B, N, T, F]
        B, N, T, nF = x.shape
        # gnn_out ---> [B,N,T,H]
        # 空间特征提取
        gnn_out = self.gnn(x,A_list)
        # 时间特征提取
        lstm_out = self.lstm(x)  # lstm_out----> [B,N,T,H]
        h = torch.cat((lstm_out,gnn_out),dim=-1) # h  ---> [B,N,T,H*2]
        h = self.dense(h)   # out---> [B,N,self.pred_len*self.ny]
        out = self.tft_decoder(h)  # [B, N, pred_len, ny]
        return out



