# OFDM simulation with 16QAM and multipath fading channel. 
# The BER performance under different SNR levels is plotted.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子，保证结果可复现

## System parameters
N_sub = 64          # Number of subcarriers
Cp_len = 16         # Length of cyclic prefix
Order = 16          # 16QAM modulation
Bps = 4             # 4 bits per symbol (16QAM)
SNR_dB_range = range(5, 21)  # SNR range: 5~20 dB (16QAM needs higher SNR)
N_symbols = 1000    # Number of OFDM symbols
num_iter = 10       # Number of iterations per SNR (for stable results)

# Multipath channel parameters (power normalized)
num_paths = 3
delay = np.array([0, 2, 4])          # Delay in samples
attenuation = np.array([1, 0.5, 0.25])
attenuation = attenuation / np.linalg.norm(attenuation)  # Normalize total path power to 1


## Transmitter function (16QAM)
def transmitter():
    # Generate random bits
    total_bits = N_sub * Bps * N_symbols
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
    
    # OFDM modulation: IFFT + cyclic prefix
    Tx_blocks = Tx_symbols.reshape(N_symbols, N_sub)
    Tx_time = np.fft.ifft(Tx_blocks, axis=1)
    Tx_time = np.hstack((Tx_time[:, -Cp_len:], Tx_time))  # Add CP
    Tx_signal = Tx_time.flatten()
    
    return Tx_signal, Tx_bits, bit_quad


## Channel function (unchanged, with multipath + AWGN)
def channel(Tx_signal, SNR_dB):
    # Add multipath fading
    faded_signal = np.zeros_like(Tx_signal, dtype=complex)
    for i in range(num_paths):
        delayed_signal = np.zeros_like(Tx_signal, dtype=complex)
        delayed_signal[delay[i]:] = Tx_signal[:-delay[i]] if delay[i] > 0 else Tx_signal
        faded_signal += attenuation[i] * delayed_signal
    
    # Add AWGN (complex noise, real/imaginary parts each have noise_power/2)
    signal_power = np.mean(np.abs(faded_signal)**2)
    SNR_linear = 10**(SNR_dB/10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(faded_signal)) + 1j * np.random.randn(len(faded_signal)))
    Rx_signal = faded_signal + noise
    
    return Rx_signal


## Receiver function (16QAM demodulation)
def receiver(Rx_signal):
    # OFDM demodulation: remove CP + FFT
    Rx_time = Rx_signal.reshape(N_symbols, N_sub + Cp_len)
    Rx_time = Rx_time[:, Cp_len:]  # Remove CP
    Rx_blocks = np.fft.fft(Rx_time, axis=1)
    Rx_symbols = Rx_blocks.flatten()
    
    # Unnormalize power (multiply by sqrt(5))
    Rx_symbols_scaled = Rx_symbols * np.sqrt(5)
    
    # Threshold-based amplitude demodulation for I/Q
    def demodulate_amplitude(x):
        amp = np.zeros_like(x, dtype=int)
        amp[x > 2] = 3
        amp[(x > 0) & (x <= 2)] = 1
        amp[(x > -2) & (x <= 0)] = -1
        amp[x <= -2] = -3
        return amp
    
    I_hat = demodulate_amplitude(np.real(Rx_symbols_scaled))
    Q_hat = demodulate_amplitude(np.imag(Rx_symbols_scaled))
    
    # Map amplitude back to Gray code bits
    amp_to_gray = {-3: [0,0], -1: [0,1], 1: [1,1], 3: [1,0]}
    I_bits = np.array([amp_to_gray[amp] for amp in I_hat])
    Q_bits = np.array([amp_to_gray[amp] for amp in Q_hat])
    
    # Combine I and Q bits to get 4-bit symbols
    Rx_bits = np.hstack((I_bits, Q_bits))
    
    return Rx_bits


## Main simulation
BER_list = []
SER_list = []

print("--- OFDM Simulation with 16QAM and Multipath Fading ---")
print(f"Subcarriers: {N_sub}, CP Length: {Cp_len}, Modulation: 16QAM")
print(f"Multipath: {num_paths} paths, Delay: {delay}, Attenuation (normalized): {attenuation}")
print(f"SNR Range: {SNR_dB_range.start}~{SNR_dB_range.stop-1} dB, Iterations per SNR: {num_iter}")
print()

for SNR_dB in SNR_dB_range:
    ber_temp, ser_temp = [], []
    for _ in range(num_iter):
        # Transmit -> Channel -> Receive
        Tx_signal, Tx_bits, Tx_bit_quad = transmitter()
        Rx_signal = channel(Tx_signal, SNR_dB)
        Rx_bits = receiver(Rx_signal)
        
        # Calculate BER and SER
        error_bits = np.sum(Rx_bits != Tx_bit_quad)
        error_symbols = np.sum(np.any(Rx_bits != Tx_bit_quad, axis=1))
        total_bits = N_sub * Bps * N_symbols
        total_symbols = N_symbols * N_sub
        
        ber_temp.append(error_bits / total_bits)
        ser_temp.append(error_symbols / total_symbols)
    
    # Average over iterations
    avg_BER = np.mean(ber_temp)
    avg_SER = np.mean(ser_temp)
    BER_list.append(avg_BER)
    SER_list.append(avg_SER)
    
    print(f"SNR: {SNR_dB:2d} dB | Avg BER: {avg_BER:.6f} | Avg SER: {avg_SER:.6f}")


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