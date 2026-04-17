import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import add_self_loops
def load_timeseries(dict_data, chem_site, chem_length):
    """Load data from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = pd.read_csv(path, delimiter=",").to_numpy()
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (chem_site, chem_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)

def to_scalar(value):
    if isinstance(value, (list, np.ndarray)):
        return value[0]
    return value


def preprocess_dynamic_data(data, train_end,val_end, log_indices=None):
    """
    处理动态数据 (X, Y)
    切分 Train/Val
    标准化 (只Fit Train)
    """
    data_processed = data.copy()

    # 1. Log Transform (针对流量、降雨等长尾分布)
    if log_indices is not None:
        for idx in log_indices:
            # 加上 epsilon 防止 log(0)
            data_processed[:, :, idx] = np.log1p(data_processed[:, :, idx])

    # 2. Split
    train_data = data_processed[:, :train_end, :]
    val_data = data_processed[:, train_end:train_end + val_end , :]
    test_data = data_processed[:, train_end + val_end:, :]
    # 3. Fit Standard Scaler (Global: across sites and time)
    # 计算 Mean/Std: 形状为 [1, 1, F]
    mean = np.nanmean(train_data, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_data, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0  # 避免除零



    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std
    return train_norm, val_norm,test_norm, mean, std

def preprocess_static_data(data, num_time_steps, log_indices=None):

    data_processed = data.copy()

    if log_indices is not None:
        for idx in log_indices:
            data_processed[:, idx] = np.log1p(data_processed[:, idx])

    # 2. Fit Standard Scaler (Global: across sites)
    # [1, F]
    mean = np.nanmean(data_processed, axis=0, keepdims=True)
    std = np.nanstd(data_processed, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    # 3. Transform
    data_norm = (data_processed - mean) / std
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    # 4. Expand & Repeat [N, F] -> [N, T, F]
    c_tensor = torch.from_numpy(data_norm).float()  # [N, F]
    c_expanded = c_tensor.unsqueeze(1)  # [N, 1, F]
    c_long = c_expanded.repeat(1, num_time_steps, 1)  # [N, T, F]

    return c_long.numpy()

def Time_emb(full_date_range):

    date_processing = pd.DataFrame(index=full_date_range)


    day_of_year = date_processing.index.dayofyear
    days_in_year = np.where(date_processing.index.is_leap_year, 366, 365)

    # --- 2. 年内周期 (Seasonality) ---
    date_processing['sin_doy'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    date_processing['cos_doy'] = np.cos(2 * np.pi * day_of_year / days_in_year)

    # --- 3. 周内周期 (Weekly Periodicity) ---

    day_of_week = date_processing.index.dayofweek
    date_processing['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7)
    date_processing['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7)

    return date_processing



def edge_extract(path,num_sites):
    edges_info = pd.read_csv(path)
    edges_weight = torch.tensor(edges_info['weight'].to_numpy(),dtype=torch.float32)

    edges_weight = torch.ones(edges_weight.shape, dtype=torch.float32)

    edges_index = torch.tensor(edges_info.iloc[:,0:2].to_numpy(),dtype=torch.long).T
    # 添加self_loop
    edge_idx,edge_weight = add_self_loops(edges_index,
                                        edges_weight,
                                        fill_value=1.0,
                                        num_nodes=num_sites)

    return edge_idx,edge_weight


def get_valid_window_indices(Y, window_size, pred_len,step=1):
    """
    步骤 1: 扫描全量数据，找出非全空样本的起始时间索引。
    输入 Y 形状: [N, T, F]
    输出: 一个包含所有有效起始索引 t 的列表。
    """
    T = Y.shape[1]
    valid_indices = []
    drop_count = 0

    for t in range(0, T - window_size - pred_len + 1, step):
        y_window = Y[:, t + window_size : t + window_size + pred_len, :]
        if np.isnan(y_window).all():
            drop_count += 1
        else:
            valid_indices.append(t)

    print(f"检测到 {drop_count} 个全空样本需要删除。保留了 {len(valid_indices)} 个有效样本。")
    return valid_indices

class SpatioTemporalDataset(Dataset):
    """
    步骤 2: 动态数据集。在获取每个 Batch 时，实时进行滑窗截取、Mask 生成和 NaN 填充。
    """
    def __init__(self, X, Y, valid_indices, window_size,pred_len,lag_matrix, max_lag):
        # 将原始完整序列转为 Tensor 以加速运算，形状保持为 [N, T, F]
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.valid_indices = valid_indices
        self.window_size = window_size
        self.pred_len = pred_len

        self.A_tensor = self.build_a_list(torch.tensor(lag_matrix), max_lag)
    def build_a_list(self,lag_matrix, max_lag):
        A_list = []
        for k in range(max_lag + 1):
            # 转置：Target行，Source列
            A_k = (lag_matrix == k).float().t()
            if k == 0:
                A_k.fill_diagonal_(1.0)
            A_list.append(A_k)
        # [max_lag + 1, N, N]
        return torch.stack(A_list)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. 获取当前有效样本的真实起始时间 t
        t = self.valid_indices[idx]

        # 2. 动态滑窗截取 [N, H, F]
        x = self.X[:, t : t + self.window_size, :]
        y = self.Y[:, t + self.window_size : t + self.window_size + self.pred_len, :]

        if x.shape[1] != self.window_size or y.shape[1] != self.pred_len:
            raise ValueError(f"Bad sample at idx={idx}, x={x.shape}, y={y.shape}")
        # 3. 生成 Mask (Mask = 1 表示有数据，Mask = 0 表示缺失)
        mask = ~torch.isnan(y)
        mask = mask.float()

        # 4. 把 Y 里的 NaN 替换成 0
        y = torch.nan_to_num(y, nan=0.0)

        return x, y, mask,self.A_tensor

def prepare_dataloader(X, Y, valid_indices, window_size,pred_len, batch_size,lag_matrix, max_lag, shuffle=True):
    """
    步骤 3: 封装生成 DataLoader
    """
    dataset = SpatioTemporalDataset(X, Y, valid_indices, window_size,pred_len,lag_matrix, max_lag)
    A_list = dataset.A_tensor
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader,A_list
