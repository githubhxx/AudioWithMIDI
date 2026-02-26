"""
噪声调度器（用于扩散模型）
参考 Tango 项目和 DDPM 的实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal
import math
torch.pi = math.pi


class NoiseSchedule:
    """噪声调度器（用于扩散模型的前向和反向过程）"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: Literal['linear', 'cosine'] = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            num_timesteps: 扩散步数
            schedule_type: 调度类型 ('linear' 或 'cosine')
            beta_start: 起始 beta 值（线性调度）
            beta_end: 结束 beta 值（线性调度）
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            # 线性调度
            self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            # 余弦调度（通常效果更好）
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.beta = torch.clamp(self.beta, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        
        # 预计算用于采样的值
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.posterior_variance = self.beta * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (self.beta * torch.sqrt(self.alpha_bar_prev) / (1 - self.alpha_bar))
        self.posterior_mean_coef2 = ((1 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1 - self.alpha_bar))
    
    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        添加噪声到输入（前向过程）
        
        Args:
            x: 干净的潜在特征 (B, C, T)
            noise: 噪声 (B, C, T)
            t: 时间步 (B,)
        
        Returns:
            noisy_x: 添加噪声后的特征 (B, C, T)
        """
        # 确保索引张量与被索引张量在同一设备上
        device = x.device
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        t = t.to(device)

        sqrt_alpha_bar_t = sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        随机采样时间步
        
        Args:
            batch_size: 批次大小
            device: 设备
        
        Returns:
            timesteps: 时间步 (batch_size,)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> tuple:
        """
        计算后验分布的均值和方差
        
        Args:
            x_start: 起始潜在特征 (B, C, T)
            x_t: 时间步 t 的潜在特征 (B, C, T)
            t: 时间步 (B,)
        
        Returns:
            posterior_mean: 后验均值 (B, C, T)
            posterior_variance: 后验方差 (B, C, T)
        """
        device = x_start.device
        posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        posterior_variance = self.posterior_variance.to(device)
        t = t.to(device)

        posterior_mean_coef1_t = posterior_mean_coef1[t].view(-1, 1, 1)
        posterior_mean_coef2_t = posterior_mean_coef2[t].view(-1, 1, 1)
        
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        posterior_variance = posterior_variance[t].view(-1, 1, 1)
        
        return posterior_mean, posterior_variance
    
    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> tuple:
        """
        计算预测分布的均值和方差
        
        Args:
            model_output: 模型输出（预测的噪声） (B, C, T)
            x_t: 时间步 t 的潜在特征 (B, C, T)
            t: 时间步 (B,)
            clip_denoised: 是否裁剪去噪后的值
        
        Returns:
            model_mean: 预测均值 (B, C, T)
            posterior_variance: 后验方差 (B, C, T)
            model_log_variance: 预测对数方差 (B, C, T)
        """
        device = x_t.device
        alpha_bar = self.alpha_bar.to(device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        t = t.to(device)

        # 预测 x_0
        sqrt_recip_alpha_bar_t = 1.0 / torch.sqrt(alpha_bar[t].view(-1, 1, 1))
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        
        pred_x_start = sqrt_recip_alpha_bar_t * (x_t - sqrt_one_minus_alpha_bar_t * model_output)
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # 计算后验分布的均值和方差
        model_mean, posterior_variance = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance_clipped[t].view(-1, 1, 1)
    
    def sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        采样下一个时间步的潜在特征
        
        Args:
            model_output: 模型输出（预测的噪声） (B, C, T)
            x_t: 时间步 t 的潜在特征 (B, C, T)
            t: 时间步 (B,)
            noise: 噪声（如果为 None 则随机采样）
        
        Returns:
            x_prev: 时间步 t-1 的潜在特征 (B, C, T)
        """
        if noise is None:
            noise = torch.randn_like(x_t)
        
        model_mean, _, model_log_variance = self.p_mean_variance(model_output, x_t, t)
        
        # 非最后一个时间步，添加噪声
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

