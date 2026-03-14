# OFDM simulation with 16QAM and multipath fading channel. 
# Constellation digrams for Tx and Rx are shown.
# The BER performance under different SNR levels is plotted.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子，保证结果可复现

## System parameters
# OFDM parameters
N_sub = 64                   # Number of subcarriers
Cp_len = 16                  # Length of cyclic prefix
Order = 16                   # 16QAM modulation
Bps = 4                      # 4 bits per symbol (16QAM)
SNR_dB_range = range(5, 21)  # SNR range: 5~20 dB (16QAM needs higher SNR)
N_symbols = 1000             # Number of OFDM symbols
num_iter = 10                # Number of iterations per SNR (for stable results)

# Pilot signal
Pilot_interval = 8 # Pilot interval (every 8 subcarriers)
Pilot_value = 1 + 0j #
# Generate pilot subcarrier indices (avoid DC and Nyquist)
Pilot_indices = np.arange(1, N_sub, Pilot_interval)
Data_indices = np.setdiff1d(np.arange(N_sub), np.concatenate(([0, N_sub//2], Pilot_indices)))

# TDL channel parameters (power normalized)
Fs = 30.72e6        # Sampling rate (5G NR 30.72 MHz)
Ts = 1/Fs  
TDL_taps_delay = np.array([0, 30, 70]) * 1e-9        # Delay in samples
TDL_taps_power_dB = np.array([0, -7, -14])
num_paths = len(TDL_taps_delay)
delay = np.round(TDL_taps_delay / Ts).astype(int)
attenuation = 10**(TDL_taps_power_dB / 20)  # Convert power to linear scale
attenuation = attenuation / np.linalg.norm(attenuation)  # Normalize total path power to 1

## Plotting function for 16QAM constellation
def plot_constellation(ax, symbols, title, sample_size=500):

    # Sample symbols (if total points < sample_size, take all)
    sample_indices = np.random.choice(len(symbols), min(sample_size, len(symbols)), replace=False)
    sampled_symbols = symbols[sample_indices]
    
    # Plot constellation
    ax.scatter(np.real(sampled_symbols), np.imag(sampled_symbols), s=15, alpha=0.7, color='#1f77b4')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('In-phase (I)', fontsize=10)
    ax.set_ylabel('Quadrature (Q)', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.axis('equal')


## Transmitter function (16QAM)
def transmitter():

    # Generate random bits
    num_data_subcarriers = len(Data_indices)
    total_bits = num_data_subcarriers * Bps * N_symbols
    Tx_bits = np.random.randint(0, 2, total_bits)
    
    # Group bits into 4-tuples for 16QAM mapping
    bit_quad = Tx_bits.reshape(-1, 4)
    
    # Gray code mapping: 2 bits -> amplitude (-3, -1, 1, 3)
    gray_to_amp = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
    
    # Map first 2 bits to I (real), last 2 bits to Q (imaginary)
    I = np.array([gray_to_amp[tuple(bits)] for bits in bit_quad[:, :2]])
    Q = np.array([gray_to_amp[tuple(bits)] for bits in bit_quad[:, 2:]])
    
    # Generate 16QAM symbols and normalize average power to 1
    # Average power of 16QAM constellation: (1+1+9+9)/4 = 5
    Tx_symbols = (I + 1j * Q) / np.sqrt(5)
    Tx_blocks = np.zeros((N_symbols, N_sub), dtype=complex)
    Tx_symbols_reshaped = Tx_symbols.reshape(N_symbols, num_data_subcarriers)
    
    for i in range(N_symbols):
        Tx_blocks[i, Data_indices] = Tx_symbols_reshaped[i]  # 填数据
        Tx_blocks[i, Pilot_indices] = Pilot_value  # 填导频

    # OFDM modulation: IFFT + cyclic prefix
    Tx_time = np.fft.ifft(Tx_blocks, axis=1)
    Tx_time = np.hstack((Tx_time[:, -Cp_len:], Tx_time))  # Add CP
    Tx_signal = Tx_time.flatten()
    
    return Tx_signal, Tx_bits, bit_quad, Tx_symbols, Tx_blocks


## Channel function (unchanged, with multipath + AWGN)
def channel(Tx_signal, SNR_dB):
    # Add TDL block fading
    num_total = len(Tx_signal) // (N_sub + Cp_len)
    fade_coeffs_block = (np.random.randn(num_paths, num_total) + 1j * np.random.randn(num_paths, num_total)) / np.sqrt(2)
    faded_signal = np.zeros_like(Tx_signal, dtype=complex)

    for i in range(num_paths):
        delayed_signal = np.zeros_like(Tx_signal, dtype=complex)
        delayed_signal[delay[i]:] = Tx_signal[:-delay[i]] if delay[i] > 0 else Tx_signal
        fade_coeffs_sampled = np.repeat(fade_coeffs_block[i], N_sub + Cp_len)
        faded_signal += attenuation[i] * fade_coeffs_sampled * delayed_signal
    
    # Add AWGN noise
    signal_power = np.mean(np.abs(faded_signal)**2)
    SNR_linear = 10**(SNR_dB/10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(faded_signal)) + 1j * np.random.randn(len(faded_signal)))
    Rx_signal = faded_signal + noise
    
    return Rx_signal


## Receiver function (fixed 16QAM mapping)
def receiver(Rx_signal, Tx_blocks):
    # Demodulate OFDM signal: remove CP + FFT
    Rx_time = Rx_signal.reshape(N_symbols, N_sub + Cp_len)
    Rx_time = Rx_time[:, Cp_len:]  # Remove CP
    Rx_blocks = np.fft.fft(Rx_time, axis=1)
    
    # LS channel estimation (unchanged)
    H_est = np.zeros((N_symbols, N_sub), dtype=complex)
    for i in range(N_symbols):
        Rx_pilot = Rx_blocks[i, Pilot_indices]
        Tx_pilot = Tx_blocks[i, Pilot_indices]
        H_est_pilot = Rx_pilot / Tx_pilot
        # Interpolate channel estimate for all subcarriers
        H_est[i] = np.interp(np.arange(N_sub), Pilot_indices, H_est_pilot)
    
    # ZF frequency domain equalization (unchanged)
    epsilon = 1e-6 # Regularization parameter to avoid division by zero
    X_est_blocks = Rx_blocks / (H_est + epsilon)
    X_est_data = X_est_blocks[:, Data_indices].flatten()
    
    # 16QAM standard constellation points (with normalized power)
    constellation_16QAM = np.array([-3-3j, -3-1j, -3+1j, -3+3j,
                                     -1-3j, -1-1j, -1+1j, -1+3j,
                                      1-3j,  1-1j,  1+1j,  1+3j,
                                      3-3j,  3-1j,  3+1j,  3+3j])/np.sqrt(5)
    
    # Map 16QAM constellation points to 4-bit Gray codes
    idx_to_bits = {
        0:  [0,0,0,0], 1:  [0,0,0,1], 2:  [0,0,1,1], 3:  [0,0,1,0],
        4:  [0,1,0,0], 5:  [0,1,0,1], 6:  [0,1,1,1], 7:  [0,1,1,0],
        8:  [1,1,0,0], 9:  [1,1,0,1], 10: [1,1,1,1], 11: [1,1,1,0],
        12: [1,0,0,0], 13: [1,0,0,1], 14: [1,0,1,1], 15: [1,0,1,0]
    }
    
    # ML demodulation (unchanged)
    Rx_bits = []
    for i in X_est_data:
        distances = np.abs(i - constellation_16QAM)
        min_idx = np.argmin(distances)
        Rx_bits.append(idx_to_bits[min_idx])
    Rx_bits = np.array(Rx_bits).flatten()
    Rx_symbols = X_est_data

    return Rx_bits, Rx_symbols


## Main simulation
if __name__ == "__main__":
    
    # Choose SNR=10dB for constellation plot
    Sample_SNR_dB = 15
    Tx_signal, _, _, Tx_symbols, Tx_blocks = transmitter()
    Rx_signal = channel(Tx_signal, Sample_SNR_dB)
    Rx_bits, Rx_symbols = receiver(Rx_signal, Tx_blocks)

    # Plot constellation Tx_symbols
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plot_constellation(ax1, Tx_symbols, title=f"Transmitted 16QAM Constellation (SNR={Sample_SNR_dB}dB)")
    # Plot constellation Rx_symbols
    plot_constellation(ax2, Rx_symbols, title=f"Received 16QAM Constellation (SNR={Sample_SNR_dB}dB)")
    plt.tight_layout()
    plt.show()

    BER_list = []
    SER_list = []

    print("--- OFDM Simulation with 16QAM and TDL-C Fading ---")
    print(f"Subcarriers: {N_sub}, CP Length: {Cp_len}, Modulation: 16QAM")
    print(f"TDL-C Channel: {num_paths} taps, Delay (samples): {delay}, Attenuation (normalized): {attenuation}")
    print(f"SNR Range: {SNR_dB_range.start}~{SNR_dB_range.stop-1} dB, Iterations per SNR: {num_iter}")
    print()

    for SNR_dB in SNR_dB_range:
        ber_temp = []
        for _ in range(num_iter):
            # Transmit -> Channel -> Receive
            Tx_signal, Tx_bits, Tx_bit_quad, _, Tx_blocks = transmitter()
            Rx_signal = channel(Tx_signal, SNR_dB)
            Rx_bits, _ = receiver(Rx_signal, Tx_blocks)
            
            # Calculate BER and SER
            error_bits = np.sum(Rx_bits != Tx_bits)
            total_bits = len(Tx_bits)
            ber_temp.append(error_bits / total_bits)

        # Average over iterations
        avg_BER = np.mean(ber_temp)
        BER_list.append(avg_BER)
    
        print(f"SNR: {SNR_dB:2d} dB | Avg BER: {avg_BER:.6f}")


    ## Plot BER performance
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB_range, BER_list, marker='o', linewidth=2, label='BER')
    plt.semilogy()
    plt.title('OFDM System Performance (16QAM, Multipath Fading)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()