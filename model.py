import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class AttentionBlock(nn.Module):
    """
    独立的自注意力模块，包含多头注意力、Dropout、残差连接和层归一化。
    """
    def __init__(self, hidden_size, num_heads, drop_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop_rate
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x: [Batch, Seq_len, hidden_size]
        # Query, Key, Value 均来自输入 x
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)

        # 残差连接与层归一化
        return self.norm(x + attn_out)

class LSTMModel(nn.Module):
    def __init__(self, nx,ny,hidden_size,drop_rate,num_heads=4):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=2, batch_first=True, bidirectional=False)

        self.attn_block = AttentionBlock(self.hidden_size, num_heads, drop_rate)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny),
        )
    def forward(self, x):  # x: [N, T, F]

        B, N, T, _ = x.shape
        x_reshaped = x.reshape(B * N, T, -1)
        x_in = F.relu(self.fc(x_reshaped))
        lstm_out,_ = self.lstm(x_in)
        attn_out = self.attn_block(lstm_out)
        mlp_out = self.mlp(attn_out)

        return mlp_out.reshape(B, N, T, self.ny)



class STGNNModel(nn.Module):
    def __init__(self, nx,ny,num_nodes,edge_index, hidden_size,drop_rate,device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.num_nodes = num_nodes
        self.device = device
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        # -------------------------------------------------------
        # 将稀疏的 edge_index 转为稠密矩阵 [N, N]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)
        adj = adj + torch.eye(num_nodes).to(device)
        deg = adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_adj = deg_inv.view(-1, 1) * adj
        self.register_buffer('norm_adj', norm_adj)

        # -------------------------------------------------------
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=2,batch_first=True)
        # GCN 公式: A_hat * X * W
        self.W1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.b1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(self.hidden_size))

        # -------------------------------------------------------

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny),
        )

        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def gcn_layer(self, x,W,b):
        # 公式: Output = Norm_Adj * X * W
        # 第一步: 聚合邻居信息 (Aggregation)
        # 利用矩阵乘法: [N, N] x [B*T, N, H] -> [B*T, N, H]
        support = torch.matmul(self.norm_adj, x)
        # 第二步: 线性变换 (Transformation)
        # [B*T, N, H] x [H, H] -> [B*T, N, H]
        gcn_out = torch.matmul(support, W) + b
        return gcn_out

    def forward(self, x):  # x: [N, T, F]

        B, N, T, _ = x.shape
        x_reshaped = x.reshape(B * N, T, -1)
        x_in = self.fc(x_reshaped)
        lstm_out, _ = self.lstm1(x_in)
        # [B*N, T, H] -> [B, N, T, H] -> [B, T, N, H] -> [B*T, N, H]
        x_gcn_in = lstm_out.view(B, N, T, self.hidden_size).permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_size)
        # 3. 执行图卷积 (并行处理 B*T 张图)
        H1 = self.gcn_layer(x_gcn_in,self.W1,self.b1)
        H1 = F.gelu(H1)
        gcn_out= self.gcn_layer(H1,self.W2,self.b2)
        # 还原维度: [B*T, N, H] -> [B, T, N, H] -> [B, N, T, H]
        out = self.mlp(gcn_out)# [B, N, T, Out]
        out = out.view(B, T, N, self.ny).permute(0, 2, 1, 3)

        return out



class GCNLayer(nn.Module):
    def __init__(self, input_dim,hidden_size,num_nodes,edge_index,drop_rate,device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size=hidden_size
        self.drop = nn.Dropout(drop_rate)
        # -------------------------------------------------------
        # 将稀疏的 edge_index 转为稠密矩阵 [N, N]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)
        adj = adj + torch.eye(num_nodes).to(device)
        deg = adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0  # 处理除0情况
        norm_adj = deg_inv.view(-1, 1) * adj
        self.register_buffer('norm_adj', norm_adj)

        # 2. GCN 参数 (手动定义权重以支持高效的 3D 矩阵运算)
        # GCN 公式: A_hat * X * W
        self.W1 = nn.Parameter(torch.FloatTensor(self.input_dim, self.hidden_size))
        self.b1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def gcn_layer(self, x,W,b):
        # 公式: Output = Norm_Adj * X * W
        # 1. 聚合 (Aggregation): A * X
        # 使用 einsum 实现: 'nm' (adj[N,N]), 'bnc' (x[B,N,C]) -> 'bmc' (out[B,N,C])
        # 这相当于对 Batch 中的每一个样本都乘了一次 adj
        support = torch.einsum('nm,bnc->bmc', self.norm_adj, x)
        # 第二步: 线性变换 (Transformation)
        # [B*T, N, H] x [H, H] -> [B*T, N, H]
        gcn_out = torch.matmul(support, W) + b
        return gcn_out
    def forward(self, x):
        gcn_out1 = self.gcn_layer(x, self.W1, self.b1)
        gcn_out1 = F.gelu(gcn_out1)
        gcn_out1 = self.drop(gcn_out1)  # 防过拟合
        gcn_out2 = self.gcn_layer(gcn_out1, self.W2, self.b2)
        return gcn_out2

class SharedGCLSTM_Cell(nn.Module):

    def __init__(self, input_dim, hidden_size,num_nodes,edge_index,drop_rate,device):
        super(SharedGCLSTM_Cell, self).__init__()
        self.input_dim = input_dim + hidden_size
        self.hidden_size = hidden_size

        # 1. 只实例化一个双层 GCN (算力消耗大户，只跑一次)
        self.shared_gcn = GCNLayer(
            self.input_dim,
            self.hidden_size,
            num_nodes,
            edge_index,
            drop_rate,
            device)

        # 2. 用轻量级 Linear 层分发给 4 个门 (算力消耗极小)
        self.gate_fc = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=2)

        spatial_feat = self.shared_gcn(combined)

        gates = self.gate_fc(spatial_feat)
        i_gate, f_gate, o_gate, c_candidate = torch.chunk(gates, 4, dim=2)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_tilde = torch.tanh(c_candidate)

        c_next = f * c + i * c_tilde
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class GC_LSTM(nn.Module):
    def __init__(self, nx, ny, num_nodes,edge_index,hidden_size, drop_rate,device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.nx, self.hidden_size)
        self.cell = SharedGCLSTM_Cell(self.hidden_size, self.hidden_size,num_nodes,edge_index,drop_rate,device)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.hidden_size, self.ny)
        )

    def forward(self, x):
        B,N, T, _ = x.shape
        x_emb = self.fc(x)

        h = torch.zeros(B,N, self.hidden_size, device=x.device)
        c = torch.zeros(B,N, self.hidden_size, device=x.device)

        outputs = []
        for t in range(T):
            x_t = x_emb[:,:,t, :]
            h, c = self.cell(x_t, h, c)
            outputs.append(h)
        lstm_out = torch.stack(outputs, dim=2)
        lstm_out = self.drop(lstm_out)
        out = self.mlp(lstm_out)
        return out



# ==========================================
# 1. 核心工具函数：构建物理滞后邻接矩阵列表
# ==========================================
def build_adj_from_lag_matrix(lag_matrix, max_lag):
    """
    将滞后时间矩阵转换为模型可用的邻接矩阵列表 A_list
    Args:
        lag_matrix: [N, N] numpy数组. lag_matrix[u, v] = k 表示 u->v 需要 k 小时.
                    -1 代表不连通.
        max_lag: 最大滞后时间步 (例如 3)
    Returns:
        A_list: 包含 [A_0, A_1, ..., A_max] 的列表
    """
    num_nodes = lag_matrix.shape[0]
    A_list = []

    print(f"\n--- 正在构建物理图结构 (Max Lag = {max_lag}) ---")

    for k in range(max_lag + 1):
        # 初始化全 0 矩阵
        A_k = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # 找出滞后时间刚好等于 k 的连接关系
        # lag_matrix[u, v] = k，代表从 u 流到 v。
        # 在 GCN 矩阵乘法 A @ X 中，A[v, u] = 1 代表 v 聚合 u 的信息。
        sources, targets = np.where(lag_matrix == k)

        for u, v in zip(sources, targets):
            A_k[v, u] = 1.0  # 转置填入：Target行，Source列

        # 强制自环：对于 A_0，确保每个节点连接自己
        if k == 0:
            diag_indices = torch.arange(num_nodes)
            A_k[diag_indices, diag_indices] = 1.0

        A_list.append(A_k)
        print(f"  -> A_{k} (滞后{k}步): 包含 {int(A_k.sum().item())} 条边")

    return A_list

# ==========================================
# 2. 核心层：物理启发的滞后图卷积 (Physics-Guided GCN)
# ==========================================
class PhysicsGuided_GCN(nn.Module):
    def __init__(self, in_features, hidden_size, A_list):
        super(PhysicsGuided_GCN, self).__init__()

        self.max_lag = len(A_list) - 1
        self.hidden_size = hidden_size


        self.norm_A_list = nn.ParameterList()
        for i, A in enumerate(A_list):
            # 计算度矩阵 D^-1
            deg = A.sum(dim=1)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            # 归一化: D^-1 * A * D^-1
            norm_A = deg_inv.view(-1, 1) * A

            # 注册为Buffer (不更新梯度，随模型保存，自动处理device)
            self.register_buffer(f'norm_A_{i}', norm_A)
            self.norm_A_list.append(norm_A)

        # 特征变换层 (共享权重)
        self.W = nn.Linear(in_features, hidden_size)

    def forward(self, x):
        """
        x: [Batch, N, T, F]
        """
        B, N, T, F_in = x.shape

        # 调整维度以适应 einsum: [B, T, N, F]
        x_trans = x.permute(0, 2, 1, 3)

        # 初始化输出容器 [B, T, N, F]
        out_agg = torch.zeros(B, T, N, F_in, device=x.device)

        for lag in range(self.max_lag + 1):
            norm_A_k = getattr(self, f'norm_A_{lag}')

            # 1. 时间平移 (Time Shifting)
            if lag == 0:
                x_lagged = x_trans
            else:
                x_lagged = torch.roll(x_trans, shifts=lag, dims=1)
                x_lagged[:, :lag, :, :] = 0.0

            '''
            2. 空间聚合 (Spatial Aggregation)
            norm_A_k: [N, N]
            x_lagged: [B, T, N, F]
            结果: [B, T, N, F] (为每个节点聚合它上游 lag 时刻发出的水)
            N = 节点数
            i = 目标节点
            j = 邻居节点
            A = 邻接矩阵
            提取邻居节点的特征，赋给目标节点，此处的邻居节点特征是上一时刻的邻居节点特征
            '''
            agg = torch.einsum('ij,btjf->btif', norm_A_k, x_lagged)

            # 累加不同滞后的影响，累加后的含义为：当前时刻节点的特征由 当前时刻节点自身的特征+上一时刻上游节点的特征
            # 源头节点的特征 只有本身的特征
            out_agg += agg

        # 变回 [B, N, T, F]
        out_final = out_agg.permute(0, 2, 1, 3)
        return F.gelu(self.W(out_final))


class PG_STGNN(nn.Module):
    def __init__(self, nx,ny,  hidden_size, lag_matrix, max_lag):
        super(PG_STGNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        A_list = build_adj_from_lag_matrix(lag_matrix, max_lag)

        self.fc = nn.Linear(self.nx, self.hidden_size )
        self.physics_gcn =nn.Sequential(
            PhysicsGuided_GCN(self.ny, self.hidden_size, A_list),
            PhysicsGuided_GCN(self.hidden_size, self.hidden_size, A_list),
            )
        self.lstm = nn.LSTM(self.hidden_size , self.hidden_size , num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size , self.ny)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size , self.hidden_size ),
            nn.ReLU(),
            nn.Linear(self.hidden_size , self.ny)
        )

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, nF = x.shape
        # --- Step 1: LSTM 时序建模 ---
        # 变性为 [B*N, T, Hidden] 让每个站点独立跑 LSTM
        x_reshaped = x.reshape(B * N, T, -1)
        T_in = self.fc(x_reshaped)
        lstm_out, (h,c) = self.lstm(T_in)

        # --- Step 2: 物理图卷积 ---
        # out: [B, N, T, Hidden]
        S_in = F.relu(self.fc1(lstm_out.reshape(B, N,T, -1)))

        h_new = self.physics_gcn(S_in)
        gnn_out = F.relu(h_new)

        h_lstm = lstm_out.reshape(B,N,T, -1)

        # #  --- 调整时空顺序 ---
        # S_in = F.relu(self.fc(x))
        # h = S_in
        # for gcn in self.physics_gcn:
        #     h_new = gcn(h)
        #     h = h + F.relu(h_new)
        # gnn_out = h
        # T_in = gnn_out.reshape(B*N, T, -1)
        # lstm_out, (_,c) = self.lstm(T_in)
        # h_lstm = lstm_out.reshape(B, N, T, -1)
        # combined = torch.cat([h_lstm, gnn_out], dim=-1)
        out = self.mlp(gnn_out)

        return out.reshape(B, N, T, -1)

class ATTPG_STGNN(nn.Module):
    def __init__(self, nx,ny,  hidden_size, lag_matrix, max_lag):
        super(ATTPG_STGNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        A_list = build_adj_from_lag_matrix(lag_matrix, max_lag)

        self.fc = nn.Linear(self.nx, self.hidden_size )
        self.physics_gcn =PhysicsGuided_GCN(self.hidden_size, self.hidden_size, A_list)
        self.lstm = nn.LSTM(self.hidden_size , self.hidden_size , num_layers=2, batch_first=True)
        self.attn_block = AttentionBlock(self.hidden_size, 4, 0.2)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size*2 , self.hidden_size ),
            nn.ReLU(),
            nn.Linear(self.hidden_size , self.ny)
        )

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, nF = x.shape
        S_in = F.relu(self.fc(x))
        gnn_out = self.physics_gcn(S_in)
        T_in = gnn_out.reshape(B*N, T, -1)
        lstm_out, (_,c) = self.lstm(T_in)
        attn_out = self.attn_block(lstm_out)
        h_lstm = attn_out.reshape(B, N, T, -1)
        combined = torch.cat([h_lstm, gnn_out], dim=-1)
        out = self.mlp(combined)

        return out.reshape(B, N, T, -1)

