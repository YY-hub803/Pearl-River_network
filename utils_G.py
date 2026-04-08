import torch
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data,Dataset
from torch_geometric.loader import DataLoader


class GraphDataset(Dataset):
    """
        全图数据集。
        输入 X: [Num_Sites, Total_Time, Feat_X]
        输入 Y: [Num_Sites, Total_Time, Feat_Y]

        Item Output:
            x_window: [History_Len, Num_Sites, Feat_X]
            y_window: [History_Len, Num_Sites, Feat_Y] (Training target reconstruction)

    """
    def __init__(self, x,y,edge,weight,H,start_ind=0, end_ind=None, min_valid_nodes=5):
        # 1. 将 numpy 数组转换为 PyTorch 张量
        self.T = x.shape[1]
        self.weight = weight
        self.H = H
        self.X = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

        self.edge_index = edge

        self.n_dataset = self.T

        total_time = x.shape[1]
        if end_ind is None:
            end_ind = total_time

        self.valid_indices = []
        print(f"正在构建数据集 (Range: {start_ind} -> {end_ind})...")
        valid_count = 0
        for i in range(start_ind, end_ind - H):

            y_window = self.y[:, i: i + H, :]
            num_valid = (~torch.isnan(y_window)).sum().item()

            if num_valid >= min_valid_nodes:

                self.valid_indices.append(i)
                valid_count += 1
        print(f"  - 原始时间步: {end_ind - start_ind - H}")
        print(f"  - 过滤后有效样本数: {len(self.valid_indices)}")

    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):

            return Subset(self, range(len(self))[index])

        elif isinstance(index, int):
            real_idx = self.valid_indices[index]

            x = self.X[:,real_idx:real_idx+self.H,:]
            y = self.y[:,real_idx:real_idx+self.H,:]
            return Data(x=x, y=y,edge_index=self.edge_index,edge_attr=self.weight)


class GraphDataset_c(Dataset):
    """
    构建的dataset包含时间窗口，dataset中每个元素都是[num_sites,windows,num_features],[num_sites,windows,num_features],edge_index
    在使用DataLoader加载数据之后，每个Batch的元素为[Batch_size*num_sites,windows,num_features],Batch_size应为num_sites的倍数。
    """
    def __init__(self, x,y,z,edge,weight,H,start_ind=0, end_ind=None, min_valid_nodes=5):
        # 1. 将 numpy 数组转换为 PyTorch 张量
        self.T = x.shape[1]
        self.weight = weight
        self.H = H
        self.X = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)
        self.z = torch.from_numpy(z).to(torch.float32)
        self.edge_index = edge

        self.n_dataset = self.T

        total_time = x.shape[1]
        if end_ind is None:
            end_ind = total_time

        self.valid_indices = []
        print(f"正在构建数据集 (Range: {start_ind} -> {end_ind})...")
        valid_count = 0
        for i in range(start_ind, end_ind - H):

            y_window = self.y[:, i: i + H, :]
            num_valid = (~torch.isnan(y_window)).sum().item()

            if num_valid >= min_valid_nodes:

                self.valid_indices.append(i)
                valid_count += 1
        print(f"  - 原始时间步: {end_ind - start_ind - H}")
        print(f"  - 过滤后有效样本数: {len(self.valid_indices)}")

    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):

            return Subset(self, range(len(self))[index])

        elif isinstance(index, int):
            real_idx = self.valid_indices[index]

            x = self.X[:,real_idx:real_idx+self.H,:]
            y = self.y[:,real_idx:real_idx+self.H,:]
            z = self.z[:,real_idx:real_idx+self.H,:]
            return Data(x=x, y=y,z=z,edge_index=self.edge_index,edge_attr=self.weight)

def created_snapshot(data,edge,attr):
    x = data

    All_dataset = list()
    for i in range(x.shape[1]):
        x_t = x[:,i,:]
        dataset = Data(x=x_t ,edge_index=edge,edge_attr=attr)
        All_dataset.append(dataset)
    return All_dataset


def edge_extract(path,num_sites):
    edges_info = pd.read_csv(path)
    edges_weight = torch.tensor(edges_info['weight'].to_numpy(),dtype=torch.float32)
    # if edges_weight.max() > 0:
    #     edges_weight = edges_weight / edges_weight.max()
    #
    edges_weight = torch.ones(edges_weight.shape, dtype=torch.float32)

    edges_index = torch.tensor(edges_info.iloc[:,0:2].to_numpy(),dtype=torch.long).T
    # 添加self_loop
    # 自环边的权重1
    edge_idx,edge_weight = add_self_loops(edges_index,
                                           edges_weight,
                                           fill_value=1.0, # 自环权重
                                           num_nodes=num_sites)

    return edge_idx,edge_weight


def get_loader(train_x, train_y,val_x, val_y,edge_index, edge_attr, batch_size, history_len):

    train_ds = GraphDataset(train_x, train_y, edge_index, edge_attr, history_len)
    val_ds = GraphDataset(val_x, val_y, edge_index, edge_attr, history_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader


def get_loader_c(train_x, train_y,val_x, val_y,train_z,val_z,edge_index, edge_attr, batch_size, history_len):

    train_ds = GraphDataset_c(train_x, train_y, train_z,edge_index, edge_attr, history_len)
    val_ds = GraphDataset_c(val_x, val_y, val_z,edge_index, edge_attr, history_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader