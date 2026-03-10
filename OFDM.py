import numpy as np
import matplotlib.pyplot as plt


## System parameters
N_sub = 64 # Number of subcarriers
Cp_len = 16 # Length of cyclic prefix
Order = 4 # QPSK-4 symbols
Bps = 2 # QPSK 2bit per symbol
SNR_dB = 15 # SNR in dB
N_symbols = 1000 # Number of symbols


## Transmitter
# generating random bits as signal
total_bits = N_sub * Bps * N_symbols
Tx_bits = np.random.randint(0, 2, total_bits)
bit_pair = Tx_bits.reshape(-1,2) # Group bits into pairs for QPSK mapping

# QPSK mapping
qpsk_map = {(0,0): -1 - 1j, (0,1): -1 + 1j, (1,0): 1 - 1j, (1,1): 1 + 1j}
# Map bits to QPSK symbols
Tx_symbols = np.array([qpsk_map[tuple(i)] for i in bit_pair])
# Normalize symbols power
Tx_symbols = Tx_symbols / np.sqrt(2)
# Group symbols into blocks
Tx_blocks = Tx_symbols.reshape(N_symbols, N_sub)
# Convert blocks to time domain
Tx_time = np.fft.ifft(Tx_blocks, axis=1)
# Add cyclic prefix
Tx_time = np.hstack((Tx_time[:, -Cp_len:], Tx_time))
# Serialize time domain signal
Tx_signal = Tx_time.flatten()


## Channel
# Add AWGN noise and fading
signal_power = np.mean(np.abs(Tx_signal)**2)
SNR_linear = 10**(SNR_dB/10)
noise_power = signal_power / SNR_linear
noise = np.sqrt(noise_power/2) * (np.random.randn(len(Tx_signal)) + 1j * np.random.randn(len(Tx_signal)))
# Add multipath fading
h = np.sqrt(0.5) * (np.random.randn(N_sub) + 1j * np.random.randn(N_sub))
print(Tx_signal.shape, h.shape)

# Apply channel
Rx_signal = Tx_signal * h + noise




