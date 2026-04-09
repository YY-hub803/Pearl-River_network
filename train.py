import numpy as np
import pandas as pd
import crit
import os
import time
import torch
from torch.cuda.amp import GradScaler, autocast
import Visualization

scaler = GradScaler()

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, weights_only=False)
    return model


def train_G(model, Train,Val, criterion, num_epochs, device,saveFolder,warmup_epochs,base_lr):

    model = model.to(device)
    criterion = criterion.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=base_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',  factor=0.5, patience=5,verbose=True,min_lr=1e-6)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    model_name = model.__class__.__name__
    lossFun_name = criterion.__class__.__name__

    if saveFolder is not None:
        if not os.path.isdir(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, f'run_printLoss.csv')
        rf = open(runFile, 'w+')

    pltRMSE_train = []
    pltRMSE_val = []

    # 早停机制
    early_stop_counter = 0
    early_stop_patience = 10  # 连续 5 个 epoch 无提升就停
    min_delta = 1e-4
    best_val_loss = float('inf')

    print(f"\n--- 开始训练 {model_name} 模型 ({device}) ---")
    for epoch in range(1,num_epochs+1):

        t0 = time.time()
        # ======== Warmup 调整学习率 ========
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optim.param_groups:
                param_group['lr'] = warmup_lr


        model.train()
        total_train_loss = 0

        for batch_X, batch_Y, batch_Mask,batch_adj in Train:

            x = batch_X.to(device)
            y = batch_Y.to(device)
            mask = batch_Mask.to(device)
            A_list = batch_adj.to(device)

            optim.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                if model_name in ("PhysicsSTGNN"):
                    outputs = model(x,A_list)
                elif model_name in ("LSTMModel","STGNNModel"):
                    outputs = model(x)
                loss = criterion(outputs, y,mask)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / len(Train)

        #----------------------------------------------------------------------------------#

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                for batch_X, batch_Y, batch_Mask,batch_adj in Val:

                    x = batch_X.to(device)
                    y = batch_Y.to(device)
                    mask = batch_Mask.to(device)
                    A_list = batch_adj.to(device)

                    if model_name in ("PhysicsSTGNN",):
                        outputs = model(x, A_list)
                    elif model_name in ("LSTMModel","STGNNModel"):
                        outputs = model(x)
                    loss_test = criterion(outputs, y,mask)
                    total_val_loss = total_val_loss + loss_test.item()

            avg_val_loss = total_val_loss / len(Val)

            pltRMSE_train.append([epoch, avg_train_loss])
            pltRMSE_val.append([epoch, avg_val_loss])

            if epoch >= warmup_epochs:
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0

                    # 可选：保存最优模型
                    if saveFolder is not None:
                        modelFile = os.path.join(saveFolder, 'best_model.pt')
                        torch.save(model, modelFile)
                        print(f"    >>> [New Best] Saved model_best.pt (Loss: {best_val_loss:.4f})")
                else:
                    early_stop_counter += 1
                    print(f"EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")

                    if early_stop_counter >= early_stop_patience:
                        print(f"\n 验证集 loss 连续 {early_stop_patience} 个 epoch 未下降，提前停止训练")
                        break
            current_lr = optim.param_groups[0]['lr']
            if current_lr < 1.1e-6 and early_stop_counter >= 3:
                print(f"\nSTOP: 学习率已降至最低 ({current_lr}) 且 Loss 无提升，提前结束。")
                break

        # printing loss
        logStr = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))
        logStr_screen = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))

        print(logStr_screen)
        if saveFolder is not None:
            rf.write(logStr + '\n')

    if saveFolder is not None:
        rf.close()
        Visualization.visualize_loss(saveFolder,lossFun_name)
    return model


def Interpolation(model,x,y,A_list,y_mean,y_std,sites_ID,saveFolder,Target_Name,device,window_size, batch_size):

    model.eval()
    model_name = model.__class__.__name__

    if saveFolder is not None:
        runFile = os.path.join(saveFolder, f'{model_name}_perform.csv')
        rf = open(runFile, 'w')

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    N_nodes, T_total, n_features = x.shape

    out_dim = model.ny
    print(f"启动高精度插补模式... 总时长: {T_total}, 窗口: {window_size}, 步长: 1")
    # --- 滑窗集成预测 (Sliding Window Loop) ---
    # 初始化累加器 (N, T, Out)
    prediction_sum = np.zeros((N_nodes, T_total, out_dim))
    prediction_counts = np.zeros((N_nodes, T_total, out_dim))
    # 计算需要滑动的总步数
    total_steps = T_total - window_size + 1
    # 生成所有窗口的起始索引
    start_indices = np.arange(0, total_steps, 1)  # step=1 for max accuracy
    total_batches = (len(start_indices) + batch_size - 1) // batch_size
    print(f"开始滑动预测... 总窗口数: {len(start_indices)}, 总 Batch 数: {total_batches}")

    with torch.no_grad():

        for batch_idx, i in enumerate(range(0, len(start_indices), batch_size)):
            batch_starts = start_indices[i: i + batch_size]

            # 2.1 构建 Batch 数据
            x_batch_list = []
            for start in batch_starts:
                end = start + window_size
                # 切片: [N, window, F]
                x_batch_list.append(x[:, start:end, :])

            # 堆叠 -> [Batch, N, window, F]
            x_batch_tensor = torch.tensor(np.array(x_batch_list), dtype=torch.float32).to(device)
            A_list_batch = A_list.unsqueeze(0).expand(len(batch_starts), -1,-1,-1)
            # 2.2 模型推理
            # output shape: [Batch, N, window, Out]
            if model_name in ("PhysicsSTGNN"):
                batch_preds = model(x_batch_tensor, A_list_batch)
            elif model_name in ("LSTMModel","STGNNModel"):
                batch_preds = model(x_batch_tensor)
            batch_preds = batch_preds.detach().cpu().numpy()

            # 2.3 累加结果 (Aggregation)
            for j, start in enumerate(batch_starts):
                end = start + window_size
                # 将当前窗口的预测值加到对应的位置
                # batch_preds[j] is [N, window, Out]
                prediction_sum[:, start:end, :] += batch_preds[j]
                prediction_counts[:, start:end, :] += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print(f"进度: Batch {batch_idx + 1}/{total_batches} 已完成...")

    print("滑动预测完成，正在计算平均值...")
    # --- 计算平均值 (Ensemble Result) ---
    # 处理边缘 (计数为0的地方设为1防止除0)
    prediction_counts[prediction_counts == 0] = 1
    final_outputs = prediction_sum / prediction_counts  # [N, T, Out]


    site_names = sites_ID["P_nm"].values if isinstance(sites_ID, pd.DataFrame) else sites_ID

    imputed_dfs = {}
    obs_dfs ={}
    for i, var_name in enumerate(Target_Name):
        print(f"\n--- 评估变量: {var_name} ---")


        pred_raw = final_outputs[:, :, i]  # [N, T]
        obs_raw = y[:, :, i]  # [N, T]

        try:
            cur_std = y_std.flat[i] if isinstance(y_std, np.ndarray) else y_std
            cur_mean = y_mean.flat[i] if isinstance(y_mean, np.ndarray) else y_mean
        except:
            cur_std = y_std[:, :, i][0][0]
            cur_mean = y_mean[:, :, i][0][0]

        pred_inv = pred_raw * cur_std + cur_mean
        obs_inv = obs_raw * cur_std + cur_mean
        pred_final = np.expm1(pred_inv)
        obs_final = np.expm1(obs_inv)

        df_pred = pd.DataFrame(pred_final, index=site_names).T
        df_obs = pd.DataFrame(obs_final, index=site_names).T

        # 核心插补逻辑：
        df_obs_clean = df_obs.replace(0, np.nan)

        imputed_dfs[var_name] = df_pred
        obs_dfs[var_name] = df_obs
        # 保存
        if saveFolder:
            filePath = saveFolder + '/out_ep' + f"{model_name}" + f'_{var_name}' + '.csv'
            if os.path.exists(filePath):
                os.remove(filePath)
            df_pred.to_csv(filePath, index=False)


        all_valid_obs = []
        all_valid_preds = []

        for site in site_names:
            # 只在有真实值的地方计算误差
            mask = (~np.isnan(df_obs_clean[site])) & (~np.isnan(df_pred[site]))

            if np.sum(mask) < 2: continue  # 数据太少跳过

            valid_obs = df_obs_clean[site][mask].values
            valid_pred = df_pred[site][mask].values

            all_valid_obs.append(valid_obs)
            all_valid_preds.append(valid_pred)

            r2 = crit.R2(valid_pred, valid_obs)
            rmse = crit.RMSE(valid_pred, valid_obs)
            nse = crit.NSE(valid_pred, valid_obs)
            kge, r, alpha, beta = crit.KGE(valid_pred, valid_obs)
            fhv = crit.FHV(valid_pred, valid_obs)

            logStr = f'Variable:{var_name}, Site:{site}, R2:{r2:.3f}, NSE:{nse:.3f},KGE:{kge:.3f},FHV:{fhv:.3f},RMSE:{rmse:.3f}'
            print(logStr)
            if rf: rf.write(logStr + '\n')
        # --- 计算整体指标 ---
        if len(all_valid_obs) > 0:
            # 将所有站点的有效数据拼接到一起
            total_obs = np.concatenate(all_valid_obs)
            total_preds = np.concatenate(all_valid_preds)

            if len(total_obs) > 0:
                total_r2 = crit.R2(total_preds, total_obs)
                total_rmse = crit.RMSE(total_preds, total_obs)

                total_nse = crit.NSE(total_preds, total_obs)
                total_kge, total_r, total_alpha, total_beta = crit.KGE(total_preds, total_obs)
                total_fhv = crit.FHV(total_preds, total_obs)

                logStr_overall = f'Variable:{var_name}, == OVERALL ==, R2:{total_r2:.3f}, NSE:{total_nse:.3f},KGE:{total_kge:.3f},FHV:{total_fhv:.3f}, RMSE:{total_rmse:.3f}'
                print(logStr_overall)
                if rf: rf.write(logStr_overall + '\n')
    if rf: rf.close()

    return imputed_dfs, obs_dfs



