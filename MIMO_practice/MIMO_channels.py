import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
 

def generate_rayleigh_channel(Nr, Nt, distance_km=0.1, carrier_freq_GHz=2.4):
    """
    生成一个考虑简化路径损耗的瑞利衰落信道矩阵。
    参数:
        Nr: 接收天线数
        Nt: 发射天线数
        distance_km: 收发距离（公里）
        carrier_freq_GHz: 载波频率（GHz）
    返回:
        H: Nr x Nt 的复信道矩阵
    """
    # 1. 生成基础瑞利衰落系数（小尺度衰落）
    # 每个元素是CN(0, 1)，即实部虚部均为独立N(0, 1/2)，总方差为1
    H_small = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
 
    # 2. 计算大尺度路径损耗（简化模型）
    # 使用自由空间路径损耗公式: PL(dB) = 20*log10(d) + 20*log10(f) + 32.44
    # 其中 d 为距离(km), f 为频率(MHz)
    distance_m = distance_km * 1000
    freq_MHz = carrier_freq_GHz * 1000
    pl_dB = 20 * np.log10(distance_m) + 20 * np.log10(freq_MHz) - 147.55  # 简化公式
    # 将dB转换为线性标度的衰减因子
    pl_linear = 10 ** (-pl_dB / 10)
 
    # 3. 合并小尺度衰落和大尺度路径损耗
    H = np.sqrt(pl_linear) * H_small
    return H

def generate_rician_channel(Nr, Nt, K_factor=3):
    """
    生成一个莱斯衰落信道矩阵。
    参数:
        Nr, Nt: 天线数
        K_factor: 莱斯K因子（线性值，非dB）
    返回:
        H: Nr x Nt 的复信道矩阵
    """
    # 确定性的直射路径分量（假设所有天线对之间的LOS相位相同，幅度为1）
    # 更复杂的模型可以考虑天线阵列的响应向量
    H_los = np.ones((Nr, Nt), dtype=complex)
 
    # 随机的散射路径分量（瑞利部分）
    H_nlos = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
 
    # 根据K因子合并
    # 总功率归一化： E{|h|^2} = 1
    # |H_los|^2 = K/(K+1), |H_nlos|^2 = 1/(K+1)
    H = np.sqrt(K_factor / (K_factor + 1)) * H_los + np.sqrt(1 / (K_factor + 1)) * H_nlos
    return H

def generate_correlated_channel(Nr, Nt, correlation_coeff=0.3):
    """
    生成具有空间相关性的MIMO信道矩阵（基于Kronecker模型）。
    参数:
        Nr, Nt: 天线数
        correlation_coeff: 相邻天线间的相关系数（0到1之间）
    返回:
        H_corr: 具有相关性的信道矩阵
    """
    # 1. 生成独立同分布的信道矩阵
    H_iid = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
 
    # 2. 构建指数衰减型相关矩阵（Toeplitz结构）
    # 发射端相关矩阵
    tx_range = np.arange(Nt)
    R_tx = correlation_coeff ** np.abs(tx_range[:, None] - tx_range[None, :])
    # 接收端相关矩阵
    rx_range = np.arange(Nr)
    R_rx = correlation_coeff ** np.abs(rx_range[:, None] - rx_range[None, :])
 
    # 3. 计算相关矩阵的平方根（Cholesky分解）
    L_tx = np.linalg.cholesky(R_tx).T  # 使得 R_tx = L_tx @ L_tx^H
    L_rx = np.linalg.cholesky(R_rx).T
 
    # 4. 应用相关性
    H_corr = L_rx @ H_iid @ L_tx.T
    return H_corr

if __name__ == "__main__":
    # 比较不同信道模型
    Nr, Nt = 4, 4
    H_rayleigh = generate_rayleigh_channel(Nr, Nt)
    H_rician = generate_rician_channel(Nr, Nt, K_factor=5)
    H_correlated = generate_correlated_channel(Nr, Nt, correlation_coeff=0.7)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im1 = axes[0].imshow(np.abs(H_rayleigh), cmap='hot', aspect='auto')
    axes[0].set_title('瑞利信道 (幅度)')
    plt.colorbar(im1, ax=axes[0])
 
    im2 = axes[1].imshow(np.abs(H_rician), cmap='hot', aspect='auto')
    axes[1].set_title('莱斯信道 (K=5, 幅度)')
    plt.colorbar(im2, ax=axes[1])
 
    im3 = axes[2].imshow(np.abs(H_correlated), cmap='hot', aspect='auto')
    axes[2].set_title('相关信道 (ρ=0.7, 幅度)')
    plt.colorbar(im3, ax=axes[2])
 
    plt.tight_layout()
    plt.show()
