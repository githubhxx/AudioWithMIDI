import numpy as np
from typing import Dict, Tuple


def _ensure_same_length(a: np.ndarray, b: np.ndarray):
    """
    将两个时间维度不同的序列裁剪到相同长度（取最短长度）。
    仅在最后一个维度上对齐，保持 (C, T) 或 (T,) 的其它维度不变。
    """
    if a.shape[-1] == b.shape[-1]:
        return a, b
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


def mel_l1_distance(mel_gen: np.ndarray, mel_ref: np.ndarray) -> float:
    """
    L1 Mel 频谱距离。
    对标 Tango/AudioLDM 中在特征空间上度量“内容接近程度”的思路，
    这里在 VAE 使用的 Mel 空间上计算一阶绝对误差。
    """
    mel_gen, mel_ref = _ensure_same_length(mel_gen, mel_ref)
    return float(np.mean(np.abs(mel_gen - mel_ref)))


def mel_l2_distance(mel_gen: np.ndarray, mel_ref: np.ndarray) -> float:
    """
    L2 Mel 频谱距离（均方误差）。
    """
    mel_gen, mel_ref = _ensure_same_length(mel_gen, mel_ref)
    return float(np.mean((mel_gen - mel_ref) ** 2))


def mel_spectral_convergence(mel_gen: np.ndarray, mel_ref: np.ndarray, eps: float = 1e-8) -> float:
    """
    简化版“谱收敛度”（Spectral Convergence），
    参考音频生成文献中常用的 STFT-SC，在 Mel 频谱上实现：

        ||M_ref - M_gen||_F / (||M_ref||_F + eps)
    """
    mel_gen, mel_ref = _ensure_same_length(mel_gen, mel_ref)
    num = np.linalg.norm(mel_ref - mel_gen)
    den = np.linalg.norm(mel_ref) + eps
    return float(num / den)


def mel_log_distance(mel_gen: np.ndarray, mel_ref: np.ndarray, eps: float = 1e-6) -> float:
    """
    对数 Mel 频谱距离（Log-Mel Distance），
    类似于语音合成中常用的 Log-STFT/Mel 损失：

        mean( |log(M_gen+eps) - log(M_ref+eps)| )
    """
    mel_gen, mel_ref = _ensure_same_length(mel_gen, mel_ref)
    log_gen = np.log(np.clip(np.abs(mel_gen), a_min=eps, a_max=None))
    log_ref = np.log(np.clip(np.abs(mel_ref), a_min=eps, a_max=None))
    return float(np.mean(np.abs(log_gen - log_ref)))


def energy_envelope_correlation(energy_gen: np.ndarray, energy_ref: np.ndarray) -> float:
    """
    能量包络相关系数，衡量整体动态和节奏起伏是否一致。
    这里的 energy 可以是：
      - 源自 STFT/Mel 的能量序列
      - 或者从波形计算得到的 RMS/能量包络
    """
    energy_gen, energy_ref = _ensure_same_length(energy_gen, energy_ref)
    # 展平成 1D
    eg = energy_gen.reshape(-1)
    er = energy_ref.reshape(-1)
    if eg.size < 2 or er.size < 2:
        return 0.0
    eg = eg - eg.mean()
    er = er - er.mean()
    num = float(np.sum(eg * er))
    den = float(np.sqrt(np.sum(eg ** 2) * np.sum(er ** 2)) + 1e-8)
    return num / den


def summarize_metrics(metric_list: Dict[str, list]) -> Dict[str, float]:
    """
    对多条样本的指标做简单平均，得到整体评估结果。
    """
    summary = {}
    for k, v in metric_list.items():
        if len(v) == 0:
            summary[k] = float("nan")
        else:
            summary[k] = float(np.mean(v))
    return summary


# ========= Fréchet Distance / FAD / KL 相关工具 =========

def compute_feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算特征的均值和协方差矩阵。

    Args:
        features: (N, D) 的特征矩阵，N 为样本数，D 为特征维度。
    Returns:
        mu:    (D,) 均值向量
        sigma: (D, D) 协方差矩阵
    """
    assert features.ndim == 2, "features 应为二维矩阵 (N, D)"
    mu = np.mean(features, axis=0)
    # rowvar=False -> 每一列视为一个变量
    sigma = np.cov(features, rowvar=False)
    # 处理协方差矩阵数值不稳定的情况（确保对称）
    sigma = (sigma + sigma.T) * 0.5
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    计算 Fréchet Distance（FD/FAD 的数学形式相同）。
    参考 FID/FAD 标准公式：

        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))

    这里只在特征空间上实现数学公式，不绑定具体嵌入模型。
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "均值向量维度不一致"
    assert sigma1.shape == sigma2.shape, "协方差矩阵维度不一致"

    diff = mu1 - mu2

    # 为避免数值问题，加入一个小的对角线抖动
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 = sigma1 + offset
    sigma2 = sigma2 + offset

    # 使用特征分解近似 sqrtm(sigma1 * sigma2)
    cov_prod = sigma1.dot(sigma2)
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    # 只取非负特征值
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    tr_covmean = np.trace(covmean)
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fd)


def frechet_distance_from_features(
    feats_gen: np.ndarray,
    feats_ref: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    直接从两组特征样本计算 Fréchet Distance。

    Args:
        feats_gen: (N_g, D) 生成样本的特征
        feats_ref: (N_r, D) 参考样本的特征
    """
    mu_g, sigma_g = compute_feature_stats(feats_gen)
    mu_r, sigma_r = compute_feature_stats(feats_ref)
    return frechet_distance(mu_g, sigma_g, mu_r, sigma_r, eps=eps)


def gaussian_kl_divergence(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    多元高斯分布之间的 KL 散度：
        KL(P || Q) where P ~ N(mu_p, sigma_p), Q ~ N(mu_q, sigma_q)

    KL(P||Q) = 0.5 * [ tr(Sigma_Q^{-1} Sigma_P)
                      + (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P)
                      - k
                      + log( det(Sigma_Q) / det(Sigma_P) ) ]

    这里同样只在特征空间上实现数学公式。
    """
    mu_p = np.atleast_1d(mu_p)
    mu_q = np.atleast_1d(mu_q)
    sigma_p = np.atleast_2d(sigma_p)
    sigma_q = np.atleast_2d(sigma_q)

    assert mu_p.shape == mu_q.shape, "均值向量维度不一致"
    assert sigma_p.shape == sigma_q.shape, "协方差矩阵维度不一致"

    k = mu_p.shape[0]

    offset = np.eye(k) * eps
    sigma_p = sigma_p + offset
    sigma_q = sigma_q + offset

    inv_sigma_q = np.linalg.inv(sigma_q)
    diff = mu_q - mu_p

    term_trace = np.trace(inv_sigma_q @ sigma_p)
    term_quad = diff.T @ inv_sigma_q @ diff
    det_sigma_p = np.linalg.det(sigma_p)
    det_sigma_q = np.linalg.det(sigma_q)
    # 防止 det 为负或 0
    det_sigma_p = max(det_sigma_p, eps)
    det_sigma_q = max(det_sigma_q, eps)
    term_logdet = np.log(det_sigma_q / det_sigma_p)

    kl = 0.5 * (term_trace + term_quad - k + term_logdet)
    return float(kl)


def gaussian_kl_from_features(
    feats_p: np.ndarray,
    feats_q: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    从两组特征样本估计高斯分布，并计算 KL(P||Q)。
    通常我们会令 P=参考分布（real），Q=生成分布（gen）。
    """
    mu_p, sigma_p = compute_feature_stats(feats_p)
    mu_q, sigma_q = compute_feature_stats(feats_q)
    return gaussian_kl_divergence(mu_p, sigma_p, mu_q, sigma_q, eps=eps)


