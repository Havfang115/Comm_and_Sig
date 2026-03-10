import numpy as np
import matplotlib.pyplot as plt


## System parameters
N_sub = 64 # Number of subcarriers
Cp_len = 16 # Length of cyclic prefix
Order = 4 # QPSK-4 symbols
Bps = 2 # QPSK 2bit per symbol
SNR_dB_range = range(5, 16) # SNR range from 5 to 15 dB
N_symbols = 1000 # Number of symbols

# Multipath channel parameters
num_paths = 3 # Number of multipath components
delay = np.array([0, 2, 4]) # Delay in samples
attenuation = np.array([1, 0.5, 0.25]) # Attenuation factors


## Transmitter function
def transmitter():
    # generating random bits as signal
    total_bits = N_sub * Bps * N_symbols
    Tx_bits = np.random.randint(0, 2, total_bits)
    
    # QPSK mapping
    bit_pair = Tx_bits.reshape(-1,2) # Group bits into pairs for QPSK mapping
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
    
    return Tx_signal, Tx_bits, bit_pair


## Channel function with multipath fading
def channel(Tx_signal, SNR_dB):
    # Add multipath fading
    faded_signal = np.zeros_like(Tx_signal, dtype=complex)
    for i in range(num_paths):
        delayed_signal = np.zeros_like(Tx_signal, dtype=complex)
        if delay[i] < len(Tx_signal):
            delayed_signal[delay[i]:] = Tx_signal[:-delay[i]] if delay[i] > 0 else Tx_signal
        faded_signal += attenuation[i] * delayed_signal
    
    # Add AWGN noise
    signal_power = np.mean(np.abs(faded_signal)**2)
    SNR_linear = 10**(SNR_dB/10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power) * (np.random.randn(len(faded_signal)) + 1j * np.random.randn(len(faded_signal)))
    Rx_signal = faded_signal + noise
    
    return Rx_signal


## Receiver function
def receiver(Rx_signal):
    # Reshape received signal into time domain blocks and remove cyclic prefix
    Rx_time = Rx_signal.reshape(N_symbols, N_sub + Cp_len)
    Rx_time = Rx_time[:, Cp_len:]
    # Convert time domain blocks to frequency domain
    Rx_blocks = np.fft.fft(Rx_time, axis=1)
    
    # Demodulation: Minimum distance demodulation
    Rx_symbols = Rx_blocks.flatten()
    Rx_bits = np.zeros((N_symbols * N_sub, 2), dtype=int)
    # Unnormalize symbols power
    Rx_bits_scaled = Rx_symbols * np.sqrt(2)
    # Demodulate bits
    Rx_bits[:,0] = (np.real(Rx_bits_scaled) > 0).astype(int)
    Rx_bits[:,1] = (np.imag(Rx_bits_scaled) > 0).astype(int)
    
    return Rx_bits


## Main simulation
BER_list = []
SER_list = []

print("--- OFDM Simulation with Multipath Fading ---")
print(f"Number of subcarriers: {N_sub}")
print(f"Cyclic prefix length: {Cp_len}")
print(f"Number of multipath components: {num_paths}")
print(f"Delay profile: {delay} samples")
print(f"Attenuation profile: {attenuation}")
print()

for SNR_dB in SNR_dB_range:
    # Transmit
    Tx_signal, Tx_bits, bit_pair = transmitter()
    
    # Channel with multipath fading and AWGN
    Rx_signal = channel(Tx_signal, SNR_dB)
    
    # Receive
    Rx_bits = receiver(Rx_signal)
    
    # Calculate BER and SER
    error_bits = np.sum(Rx_bits != bit_pair)
    error_symbols = np.sum(np.any(Rx_bits != bit_pair, axis=1))
    total_bits = N_sub * Bps * N_symbols
    BER = error_bits / total_bits
    SER = error_symbols / N_symbols
    
    BER_list.append(BER)
    SER_list.append(SER)
    
    print(f"SNR_dB: {SNR_dB}, BER: {BER:.6f}, SER: {SER:.6f}")


## Plot BER performance
plt.figure(figsize=(10, 6))
plt.plot(SNR_dB_range, BER_list, marker='o', linewidth=2, label='BER')
plt.semilogy()  # Use semilogarithmic scale for BER
plt.title('OFDM System Performance with Multipath Fading')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

