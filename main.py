import random
import numpy as np
import pandas as pd
import torch
import train
import model
import crit
import shutil
import glob
import os
import General_utils,utils_G
import Visualization as vis


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set seeds
random_seed = 40
set_seeds(random_seed)

# set GPU
if torch.cuda.is_available():
    GPUid = 0
    torch.cuda.set_device(GPUid)


hyper_params = {
    "epoch_run": 150,
    "epoch_save": 10,
    "hidden_size": 32,
    'history_len': 14,
    "batch_size":128,
    "num_layers" : 2,
    "drop_rate": 0.3,
    "warmup_epochs":10,
    "base_lr":1e-4,
    "BACKEND":"STGNNModel", # select model    STGNNModel/ LSTMModel/GC_LSTM/PG_STGNN
    "lossFun":'RMSE'
}


MODEL_FACTORY = {
    "GC_LSTM": model.GC_LSTM,
    "LSTMModel": model.LSTMModel,
    "STGNNModel": model.STGNNModel,
    "PG_STGNN":model.PG_STGNN,
    "ATTPG_STGNN":model.ATTPG_STGNN
}
Loss_FACTORY = {
    "MSE": crit.MSELoss,
    "NSE": crit.NSELoss,
    "RMSE": crit.RMSELoss,
    "MixLoss": crit.MixLoss,
    "WeightLoss":crit.WeightLoss

}



freq = '1D'
dir_model = freq+'_' + "%s_H%d_L%d_dr%.2f_NL%d_E%d" % (
    hyper_params['BACKEND'],
    hyper_params['hidden_size'],
    hyper_params['history_len'],
    hyper_params['drop_rate'],
    hyper_params["num_layers"],
    hyper_params['epoch_run'],
)
# set input and output folders
dir_proj = f"data"
work_path = os.getcwd()
dir_input = os.path.join(work_path, dir_proj)
dir_output = os.path.join("OutPut",dir_model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKEND= hyper_params["BACKEND"]

num_sites = 24 # 29
D_R = pd.read_csv(os.path.join(dir_input, 'D_R.csv'))
# os.path.join(dir_input, 'D_R_4h.csv').split('\\')[-1] ------> 'D_R_4h.csv'
# os.path.join(dir_input, 'D_R_4h.csv').split('\\')[-1].split('_')[-1]-----> '4h.csv'
# freq----> "4h"
start_date = D_R['start'].min()
end_date = D_R['end'].max()
full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
date_length = len(full_date_range)

print("---------------------划分窗格及数据集---------------------")
train_rate = 0.7
val_rate = 0.3
train_end = int(date_length * train_rate) # 划分点
val_date_range = full_date_range[train_end:,]
# train_end = date_length # 划分点

#------------------------------------- load data -----------------------------------------------------------------------
print("------------------------ load path ------------------------------")
dir_x = {
    "x_pre": os.path.join(dir_input, 'input_xforce_prcp.csv'),
    "x_pet": os.path.join(dir_input, 'input_xforce_pet.csv'),
    "x_solar":os.path.join(dir_input, 'input_xforce_solar.csv'),
    "x_temp": os.path.join(dir_input, 'input_xforce_temp.csv'),
    "x_u": os.path.join(dir_input, 'input_xforce_u_wind.csv'),
"   x_v": os.path.join(dir_input, 'input_xforce_v_wind.csv'),
    "x_vp": os.path.join(dir_input, 'input_xforce_vp.csv'),
}

dir_c = {
    "c_all": os.path.join(dir_input, 'input_c_all.csv'),
}

dir_y = {
    # "Flux": os.path.join(dir_input, 'input_yobs_Flux.csv'),
    # "DIS": os.path.join(dir_input, 'input_yobs_Dis.csv'),
    "TP": os.path.join(dir_input, 'input_yobs_Tp.csv')
}

edge_path = os.path.join(dir_input, 'edge_weight.csv')
vis_folder = os.path.join(dir_output, 'visualization')

if not os.path.exists(vis_folder):
    # 创建文件夹，如果有必要会创建中间目录
    os.makedirs(vis_folder, exist_ok=True)
    print(f"成功创建模型输出文件夹: {vis_folder}")
else:
    print(f"模型输出文件夹已存在: {vis_folder}")
    shutil.rmtree(vis_folder, ignore_errors=True)
    os.makedirs(vis_folder, exist_ok=True)


print("------------------------ load data ------------------------------")

print('  Loading X (Forcing)...')
x = General_utils.load_timeseries(dir_x, num_sites, date_length)

print('  Loading C (Static Attributes)...')
c = General_utils.load_attribute(dir_c)
c[np.where(np.isnan(c))] = 0

print('  Loading Y (Targets)...')
y = General_utils.load_timeseries(dir_y, num_sites, date_length)

print("------------------------ processing data ------------------------------")
date_processing = General_utils.Time_emb(full_date_range)
date_array = date_processing.values
date_array_expanded = np.expand_dims(date_array, axis=0)
date_emb = np.repeat(date_array_expanded, num_sites, axis=0)
train_date = date_emb[:, :train_end, :]
val_date  = date_emb[:, train_end:, :]

c_long = General_utils.preprocess_static_data(c, date_length, log_indices=None)
train_c = c_long[:, :train_end, :]
val_c   = c_long[:, train_end:, :]

# list(range(x.shape[2]))
train_x, val_x, x_mean, x_std = General_utils.preprocess_dynamic_data(
    x, train_end, log_indices=list(range(x.shape[2]))
)

train_x = np.nan_to_num(train_x, nan=0.0)
val_x = np.nan_to_num(val_x, nan=0.0)

train_x = np.concatenate([train_x, train_c], axis=2)
val_x   = np.concatenate([val_x, val_c], axis=2)

train_y, val_y, y_mean, y_std = General_utils.preprocess_dynamic_data(
    y, train_end, log_indices=list(range(y.shape[2]))
)
print(f"  Train Data Shapes: X{train_x.shape}, Y{train_y.shape}")
print(f"  Val Data Shapes:   X{val_x.shape}, Y{val_y.shape}")

print('  ------------------------loading edges_info ------------------------------')
edge,weight = utils_G.edge_extract(edge_path,num_sites)
if BACKEND in ("PG_STGNN","ATTPG_STGNN"):
    Lag_Matrix_path = os.path.join(dir_input, 'Lag_Matrix.csv')
    lag_matrix = pd.read_csv(Lag_Matrix_path, header=None)

print('  ------------------------loading sites_info ------------------------------')
sites_ID= pd.read_csv(os.path.join(dir_input,"points_info.csv"))
coords = sites_ID.iloc[:,2:4].values
print("------------------------ creating window ------------------------------")

print("Train Set:")
train_valid_indices = General_utils.get_valid_window_indices(train_y, hyper_params['history_len'])
print("Val Set:")
val_valid_indices = General_utils.get_valid_window_indices(val_y, hyper_params['history_len'])

print('  ------------------------ DataLoader ------------------------------')
Train = General_utils.prepare_dataloader(
    train_x, train_y,
    train_valid_indices,
    hyper_params['history_len'],
    hyper_params['batch_size'],
    shuffle=True)
Val = General_utils.prepare_dataloader(
    val_x, val_y,
    val_valid_indices,
    hyper_params['history_len'],
    hyper_params['batch_size'],
    shuffle=False)

nx = train_x.shape[-1]
ny = train_y.shape[-1]

if BACKEND in ("LSTMModel"):
    model = MODEL_FACTORY[BACKEND](
        nx, ny,
        hyper_params['hidden_size'],
        hyper_params['drop_rate']
    )
elif BACKEND in ("STGNNModel","GC_LSTM"):
    model = MODEL_FACTORY[BACKEND](
        nx, ny,num_sites,edge,
        hyper_params['hidden_size'],
        hyper_params['drop_rate'],
        device
    )
elif BACKEND in ("PG_STGNN","ATTPG_STGNN"):
    model = MODEL_FACTORY[BACKEND](
        nx, ny,
        hyper_params['hidden_size'],
        lag_matrix,
        2,
    )
else:
    raise ValueError(f"Unknown BACKEND type: {BACKEND}")

print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


lossFun = Loss_FACTORY[hyper_params['lossFun']]()


model_test = train.train_G(
    model, coords,Train, Val,lossFun,
    hyper_params['epoch_run'],
    device,
    dir_output,
    hyper_params['warmup_epochs'],
    hyper_params['base_lr'],
    hyper_params['epoch_save']
)

model_files = glob.glob(os.path.join(dir_output, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")
# 按照文件修改时间排序，获取最新的模型
latest_model_path = max(model_files, key=os.path.getmtime)

print(f">>> 加载原始模型进行插补: {latest_model_path}")
model_raw = torch.load(latest_model_path)
x_in = np.concatenate([train_x, val_x], axis=1)
y_in = np.concatenate([train_y, val_y], axis=1)
Target_Name = list(dir_y.keys())
y_out, y_true = train.Interpolation(
    model_raw, val_x, val_y,
    y_mean, y_std, sites_ID, dir_output, Target_Name,device,
    hyper_params['history_len'],hyper_params['batch_size']
)

# ------------------------ 可视化部分 ------------------------------
if 'y_out' in locals():
    print("------------------------ 生成可视化图表 ------------------------------")
    vis_mapping = {
        "Flux": lambda: vis.vis_filled(y_true['Flux'], y_out['Flux'], val_date_range, vis_folder, "Flux"),
        "DIS": lambda: vis.vis_filled(y_true['DIS'], y_out['DIS'], val_date_range, vis_folder, "DIS"),
        "TP": lambda: vis.vis_filled(y_true['TP'], y_out['TP'], val_date_range, vis_folder, "TP")
    }
    for var_name, vis_func in vis_mapping.items():
        if var_name in Target_Name:
            vis_func()  # 执行对应变量的可视化函数
            print(f"已执行 {var_name} 的可视化，保存至 {vis_folder}")





