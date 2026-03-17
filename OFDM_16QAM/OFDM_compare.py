# OFDM simulation with 16QAM and multipath fading channel. 
# Constellation digrams for Tx and Rx are shown.
# The BER performance under different SNR levels is plotted.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import Akima1DInterpolator

np.random.seed(42)

## System parameters (Updated for 5G NR - 20 MHz Bandwidth)
# OFDM parameters
Fs = 30.72e6                 # Sampling rate (30.72 MHz)
SCS = 15e3                   # Subcarrier Spacing (15 kHz)
N_sub = int(Fs / SCS)        # FFT Size: 2048 subcarriers
N_used = 1272                # Active subcarriers (106 Resource Blocks * 12)
Cp_len = 144                 # Length of cyclic prefix (Normal CP)
Order = 16                   # 16QAM modulation
Bps = 4                      # 4 bits per symbol (16QAM)
SNR_dB_range = range(5, 21)  
N_symbols = 500             # REDUCED from 1000! (A 2048 FFT is computationally heavy)
num_iter = 1

# Pilot signal (Mimicking 5G DMRS spacing)
Pilot_interval = 12 
Pilot_value = np.sqrt(2) * (1 + 0j) 

# Generate active subcarrier indices (using the first 1272 valid subcarriers to keep interpolation stable)
active_indices = np.arange(1, N_used + 1) 
Pilot_indices = np.arange(1, N_used + 1, Pilot_interval)
if Pilot_indices[-1] != N_used:
    Pilot_indices = np.append(Pilot_indices, N_used)
Data_indices = np.setdiff1d(active_indices, Pilot_indices)

# 5G TDL-C Channel parameters (Nominal 300ns Delay Spread)
Ts = 1/Fs  
TDL_taps_delay = np.array([0, 100, 300, 500]) * 1e-9        # Mimicking a 300ns RMS delay spread
TDL_taps_power_dB = np.array([0, -2, -6, -10])              # Realistic exponential power decay
num_paths = len(TDL_taps_delay)
delay = np.round(TDL_taps_delay / Ts).astype(int)
attenuation = 10**(TDL_taps_power_dB / 20)  
attenuation = attenuation / np.linalg.norm(attenuation)

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
    Tx_symbols = (I + 1j * Q) / np.sqrt(10)
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
    
    H_est = np.zeros((N_symbols, len(active_indices)), dtype=complex)
    noise_power_est = np.zeros(N_symbols, dtype=complex)
    for i in range(N_symbols):
        Rx_pilot = Rx_blocks[i, Pilot_indices]
        Tx_pilot = Tx_blocks[i, Pilot_indices]
        H_est_pilot = Rx_pilot / Tx_pilot
        
        akima_real = Akima1DInterpolator(Pilot_indices, np.real(H_est_pilot))
        akima_imag = Akima1DInterpolator(Pilot_indices, np.imag(H_est_pilot))
        H_real = akima_real(active_indices)
        H_imag = akima_imag(active_indices)
        H_est[i] = H_real + 1j* H_imag

        noise_est = Rx_pilot - H_est[i, Pilot_indices-1] * Tx_pilot
        noise_power_est[i] = np.mean(np.abs(noise_est)**2)

    # Estimate noise power
    N0_est = np.mean(noise_power_est)

    # MMSE equalization
    epsilon = 1e-6 # Regularization parameter to avoid division by zero
    H_conj = np.conj(H_est)
    # MMSE formula: X_est = H_conj / (|H|^2 + 1/SNR) * Rx
    MMSE_coeff = H_conj / (np.abs(H_est)**2 + N0_est + epsilon)
    X_est_blocks = Rx_blocks[:,active_indices] * MMSE_coeff
    X_est_data = X_est_blocks[:, Data_indices-1].flatten()
    
    # 16QAM standard constellation points (with normalized power)
    constellation_16QAM = np.array([-3-3j, -3-1j, -3+1j, -3+3j,
                                     -1-3j, -1-1j, -1+1j, -1+3j,
                                      1-3j,  1-1j,  1+1j,  1+3j,
                                      3-3j,  3-1j,  3+1j,  3+3j])/np.sqrt(10)
    
    # Map 16QAM constellation points to 4-bit Gray codes
    idx_to_bits = {
        0:  [0,0,0,0], 1:  [0,0,0,1], 2:  [0,0,1,1], 3:  [0,0,1,0],
        4:  [0,1,0,0], 5:  [0,1,0,1], 6:  [0,1,1,1], 7:  [0,1,1,0],
        8:  [1,1,0,0], 9:  [1,1,0,1], 10: [1,1,1,1], 11: [1,1,1,0],
        12: [1,0,0,0], 13: [1,0,0,1], 14: [1,0,1,1], 15: [1,0,1,0]
    }
    
    # ML demodulation (unchanged)
    # Vectorized distance calculation
    distances = np.abs(X_est_data[:, np.newaxis] - constellation_16QAM)
    min_indices = np.argmin(distances, axis=1)
    
    # Map back to bits
    Rx_bits = np.array([idx_to_bits[idx] for idx in min_indices]).flatten()
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