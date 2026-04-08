import torch
import torch.nn as nn
import numpy as np


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target, mask):
        """
        output: [B, N, T, F]
        target: [B, N, T, F]
        mask:   [B, N, T, F] (1=有效, 0=缺失)
        """

        # 平方误差
        loss = (output - target) ** 2

        # mask过滤
        loss = loss * mask

        # 先求MSE再开方
        mse = loss.sum() / (mask.sum() + 1e-6)
        rmse = torch.sqrt(mse)

        return rmse

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target,mask):
        """
        preds:   [Batch, N, T, F]
        targets: [Batch, N, T, F]
        mask:    [Batch, N, T, F] (1=真实值, 0=缺失)
        """
        # 1. 计算所有位置的平方差
        loss = (output - target) ** 2

        # 2. 用 Mask 过滤：只保留 mask=1 的位置的误差，其他位置变 0
        loss = loss * mask

        return loss.sum() / (mask.sum() + 1e-6)

class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target, mask):
        """
        output: [B, N, T, F]
        target: [B, N, T, F]
        mask:   [B, N, T, F]
        """

        # 有效值数量
        valid_count = mask.sum() + 1e-6

        # -------- 分子：残差平方和 --------
        numerator = ((output - target) ** 2) * mask
        numerator = numerator.sum()

        # -------- 分母：真实值方差平方和 --------
        mean_target = (target * mask).sum() / valid_count

        denominator = ((target - mean_target) ** 2) * mask
        denominator = denominator.sum() + 1e-6

        nse = 1 - numerator / denominator

        # 作为loss返回 → 越小越好
        loss = 1 - nse

        return loss


class MixLoss(nn.Module):
    def __init__(
        self,w_rmse=1.0,w_peak=0.6,w_grad=0.3,w_var=0.2,peak_alpha=4.0):
        super().__init__()

        self.w_rmse = w_rmse
        self.w_peak = w_peak
        self.w_grad = w_grad
        self.w_var  = w_var
        self.peak_alpha = peak_alpha

    def forward(self, output, target, mask):

        eps = 1e-6

        # ======================
        # 1️⃣ RMSE（主误差）
        # ======================
        sq_err = (output - target) ** 2
        mse = (sq_err * mask).sum() / (mask.sum() + eps)
        rmse = torch.sqrt(mse + eps)

        # ======================
        # 2️⃣ 峰值加权误差（防止压峰）
        # ======================
        # 按真实值大小加权
        t_max = (target * mask).max().detach() + eps
        weight = 1 + self.peak_alpha * (target / t_max)

        peak_mse = (sq_err * weight * mask).sum() / (mask.sum() + eps)

        # ======================
        # 3️⃣ Masked Gradient Loss（波动趋势）
        # 只在连续真实点之间计算
        # ======================
        d_out = output[:,:,1:,:] - output[:,:,:-1,:]
        d_tar = target[:,:,1:,:] - target[:,:,:-1,:]

        pair_mask = mask[:,:,1:,:] * mask[:,:,:-1,:]

        grad_loss = ((d_out - d_tar)**2 * pair_mask).sum() / (pair_mask.sum() + eps)

        # ======================
        # 4️⃣ 方差匹配（防止过平滑）
        # ======================
        v = mask.sum() + eps

        mean_o = (output * mask).sum() / v
        mean_t = (target * mask).sum() / v

        var_o = ((output - mean_o)**2 * mask).sum() / v
        var_t = ((target - mean_t)**2 * mask).sum() / v

        var_loss = torch.abs(var_o - var_t)

        # ======================
        # 总Loss
        # ======================
        total = (
            self.w_rmse * rmse +
            self.w_peak * peak_mse +
            self.w_grad * grad_loss +
            self.w_var  * var_loss
        )

        return total


class WeightLoss(nn.Module):
    def __init__(self, w_q=1.0, w_c=10.0):
        """
        多任务损失函数：同时优化流量(Q)和浓度(C)
        Args:
            w_q (float): 流量 Q 的损失权重 (通常设为 1.0)
            w_c (float): 浓度 C 的损失权重 (建议设大一点，如 5.0 或 10.0，因为 C 难预测且数据少)
        """
        super(WeightLoss, self).__init__()
        self.w_q = w_q
        self.w_c = w_c

    def forward(self, output, target, mask):
        """
        output: [B, N, T, 2] -> 假设第0列是Q, 第1列是C
        target: [B, N, T, 2] -> 假设第0列是Q, 第1列是C
        mask:   [B, N, T, 2] -> 1=有效数据, 0=缺失数据 (Q通常全1, C会有很多0)
        """
        eps = 1e-6

        # --- 1. 拆分变量 ---
        # 假设: dim=-1 的索引 0 是 Q (Flow), 索引 1 是 C (Conc)
        pred_q = output[..., 0]
        true_q = target[..., 0]
        mask_q = mask[..., 0]

        pred_c = output[..., 1]
        true_c = target[..., 1]
        mask_c = mask[..., 1]

        # --- 2. 计算流量 Loss (Q) ---
        # Q 通常是连续的，mask_q 大部分为 1
        sq_err_q = (pred_q - true_q) ** 2
        loss_q = (sq_err_q * mask_q).sum() / (mask_q.sum() + eps)

        # --- 3. 计算浓度 Loss (C) ---
        # C 有大量缺失，mask_c 会自动过滤掉 NaN/缺失值 (mask=0的位置)
        # 只有当 mask_c=1 时，误差才会被计入
        sq_err_c = (pred_c - true_c) ** 2
        loss_c = (sq_err_c * mask_c).sum() / (mask_c.sum() + eps)

        # --- 4. 加权总和 ---
        total_loss = self.w_q * loss_q + self.w_c * loss_c

        return total_loss

def R2(output, target):
    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """

    mask = target == target
    p0 = output[mask]
    t0 = target[mask]
    N = len(t0)
    #计算分子部分
    numerator = (N * np.sum(p0 * t0) - np.sum(p0) * np.sum(p0)) **2

    #计算分母
    denominator1 = (N * np.sum(t0**2) - (np.sum(t0))**2)
    denominator2 = (N * np.sum(p0**2) - (np.sum(p0))**2)
    denominator = denominator1*denominator2

    R2 = numerator / denominator
    return R2

def NSE(output, target):

    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]

    numerator = np.sum((t0 - p0) ** 2)
    denominator = np.sum((t0- np.average(t0))** 2)

    NSE = 1 - numerator / denominator

    return NSE

def MAE(output, target):
    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]
    N = len(t0)

    MAE = np.sum(np.abs(t0 - p0))/N
    return MAE

def RMSE(output, target):
    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]
    N = len(t0)

    RMSE = np.sqrt(np.mean(t0 - p0)**2)

    return RMSE