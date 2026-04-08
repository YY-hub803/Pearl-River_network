import numpy as np
import pandas as pd
import os
import Visualization as vis



LSTM_Path =  "Yangtze_upper_31TR_output/1D_LSTMModel_H64_dr0.10_NL1_E300"
STGNN_Path = "Yangtze_upper_31TR_output/1D_STGNNModel_H64_dr0.10_NL1_E300"
LSTM_filePath = [LSTM_Path + '/out_ep' + "LSTMModel"  + f'_y_Out{i+1}' + '.csv' for i in range(2)]
STGNN_filePath = [STGNN_Path + '/out_ep' + "STGNNModel"  + f'_y_Out{i+1}' + '.csv'for i in range(2)]
site_Path = "Yangtze_upper_31TR/R1D/points_info.txt"
observed_Path = ["Yangtze_upper_31TR/R1D/original_width_flux.csv","Yangtze_upper_31TR/R1D/original_width_tp.csv"]



Sites_Id = np.loadtxt(site_Path,dtype=str,delimiter="\t",skiprows=1,usecols=[1])
Observed_flux = pd.read_csv(observed_Path[0]).iloc[:1108]
LSTM_output_flux = pd.read_csv(LSTM_filePath[0])
STGNN_output_flux = pd.read_csv(STGNN_filePath[0])
Observed_tp = pd.read_csv(observed_Path[1]).iloc[:1108]
LSTM_output_tp = pd.read_csv(LSTM_filePath[1])
STGNN_output_tp = pd.read_csv(STGNN_filePath[1])
# total_time_sequence

total_step =  np.arange(len(Observed_tp))

combined_flux = np.hstack([Observed_flux, LSTM_output_flux, STGNN_output_flux])
column_names = [f'Observed_{i}' for i in Sites_Id] + \
               [f'STGNN_{i}' for i in Sites_Id] + \
               [f'LSTM_{i}' for i in Sites_Id]
all_result_flux = pd.DataFrame(combined_flux,
                       columns=column_names)

combined_tp = np.hstack([Observed_tp, LSTM_output_tp, STGNN_output_tp])
column_names = [f'Observed_{i}' for i in Sites_Id] + \
               [f'STGNN_{i}' for i in Sites_Id] + \
               [f'LSTM_{i}' for i in Sites_Id]
all_result_tp = pd.DataFrame(combined_tp,
                       columns=column_names)



vis_folder = 'visualization'
if not os.path.exists(vis_folder):
    # 创建文件夹，如果有必要会创建中间目录
    os.makedirs(vis_folder, exist_ok=True)
    print(f"成功创建模型输出文件夹: {vis_folder}")
else:
    print(f"模型输出文件夹已存在: {vis_folder}")

vis.box_plot(all_result_flux,Sites_Id,"Flux",vis_folder)
vis.perform_compare(all_result_flux,Sites_Id,total_step,"Flux",vis_folder)
vis.residual(all_result_flux,Sites_Id,total_step,"Flux",vis_folder)


vis.box_plot(all_result_tp,Sites_Id,"TP",vis_folder)
vis.perform_compare(all_result_tp,Sites_Id,total_step,"TP",vis_folder)
vis.residual(all_result_tp,Sites_Id,total_step,"TP",vis_folder)