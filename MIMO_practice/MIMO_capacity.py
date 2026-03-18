import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from MIMO_channel import generate_rayleigh_channel, generate_rician_channel, generate_correlated_channel

def mimo_capacity(H, snr_db):
    """
    计算给定信道矩阵H和SNR（dB）下的瞬时信道容量（bps/Hz）。
    假设发射端未知CSI，功率均匀分配。
    """
    Nr, Nt = H.shape
    # 将SNR从dB转换为线性值
    snr_linear = 10 ** (snr_db / 10.0)
 
    # 计算 H * H^H
    H_H = H.conj().T
    HH_product = H @ H_H
 
    # 单位矩阵
    I = np.eye(Nr, dtype=complex)
 
    # 核心容量公式
    # 使用 np.linalg.det 计算行列式，注意结果可能是复数，但行列式值应为实数
    det_arg = I + (snr_linear / Nt) * HH_product
    capacity = np.log2(np.linalg.det(det_arg).real)  # 取实部，理论上应为实数
 
    # 确保容量非负（数值误差可能导致微小负数）
    return max(capacity, 0)
 
def ergodic_capacity(Nr, Nt, snr_db, channel_type='rayleigh', num_realizations=1000):
    """
    通过蒙特卡洛仿真计算遍历容量。
    参数:
        Nr, Nt: 天线数
        snr_db: 信噪比(dB)
        channel_type: 信道类型 ('rayleigh', 'rician', 'correlated')
        num_realizations: 信道实现次数
    返回:
        平均容量 (bps/Hz)
    """
    capacities = []
    for _ in range(num_realizations):
        if channel_type == 'rayleigh':
            H = generate_rayleigh_channel(Nr, Nt)
        elif channel_type == 'rician':
            H = generate_rician_channel(Nr, Nt, K_factor=5)
        elif channel_type == 'correlated':
            H = generate_correlated_channel(Nr, Nt, correlation_coeff=0.5)
        else:
            raise ValueError("不支持的 channel_type")
 
        cap = mimo_capacity(H, snr_db)
        capacities.append(cap)
 
    return np.mean(capacities)

# 绘制容量 vs SNR 曲线
snr_range_db = np.arange(-10, 31, 2)  # SNR从-10dB到30dB
configs = [(2, 2), (4, 4), (8, 8)]
channel_types = ['rayleigh', 'rician', 'correlated']
colors = ['b', 'g', 'r', 'c', 'm', 'y']
 
plt.figure(figsize=(12, 8))
 
# 子图1：不同天线配置，瑞利信道
ax1 = plt.subplot(2, 2, 1)
for idx, (Nr, Nt) in enumerate(configs):
    capacities = [ergodic_capacity(Nr, Nt, snr, 'rayleigh', 500) for snr in snr_range_db]
    ax1.plot(snr_range_db, capacities, marker='o', label=f'{Nr}x{Nt}', color=colors[idx])
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('Ergodic capacity (bps/Hz)')
ax1.set_title('Capacity vs SNR for different antenna configurations (Rayleigh channel)')
ax1.grid(True, alpha=0.3)
ax1.legend()
 
# 子图2：固定4x4配置，不同信道模型
ax2 = plt.subplot(2, 2, 2)
Nr, Nt = 4, 4
for idx, chan_type in enumerate(channel_types):
    capacities = [ergodic_capacity(Nr, Nt, snr, chan_type, 500) for snr in snr_range_db]
    ax2.plot(snr_range_db, capacities, marker='s', label=chan_type, color=colors[idx])
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('Ergodic capacity (bps/Hz)')
ax2.set_title('Capacity vs SNR for 4x4 MIMO under different channel models')
ax2.grid(True, alpha=0.3)
ax2.legend()
 
# 子图3：容量随天线数增长（高SNR=20dB）
ax3 = plt.subplot(2, 2, 3)
snr_fixed = 20
antenna_counts = np.arange(1, 9)  # 从1x1到8x8
capacities_siso = []
capacities_mimo = []
for n in antenna_counts:
    # SISO 容量
    cap_siso = np.log2(1 + 10**(snr_fixed/10))
    capacities_siso.append(cap_siso)
    # NxN MIMO 容量（近似线性增长）
    cap_mimo = ergodic_capacity(n, n, snr_fixed, 'rayleigh', 300)
    capacities_mimo.append(cap_mimo)
 
ax3.plot(antenna_counts, capacities_siso, 'b--o', label='SISO (Theory)')
ax3.plot(antenna_counts, capacities_mimo, 'r-s', label='NxN MIMO (Simulation)')
ax3.set_xlabel('Number of antenna (N)')
ax3.set_ylabel(f'Capacity @ SNR={snr_fixed}dB (bps/Hz)')
ax3.set_title('Capacity trend as number of antennas grows')
ax3.grid(True, alpha=0.3)
ax3.legend()
 
plt.tight_layout()
plt.show()