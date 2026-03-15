# 16QAM OFDM System Simulation Log
## version 1.0
Log:
- Build the basic OFDM 16QAM system simulation with AWGN and 3-path multipath fading channel.
- Iterated the simulation for 10 times for each SNR level to get stable results.
- Evaluate the BER and SER performance of the system.

Thoughts:
- The BER is actually quite high the it should be.
## version 1.1
Log:
- Added the constellation diagrams for Tx and Rx signals.

Thoughts:
- Constellation diagrams are not very helpful for the case, but good for understanding the system.

## version 2.0
Log:
- Changed the simple multipath fading into 3-path TDL model.

Thoughts:
- The BER is around 0.5 no matter the value of SNR. Why? I have to adjust this.

## version 2.1
Log:
- Added channel estimation and improved the demodulation part.

Thoughts:
- The BER is decreasing as SNR increases, but the curve is not as it supposed to be. The BER rate at high SNR is still too high.

## version 2.2
Log:
- Changed to 5G standard parameters. 
- Used MMSE channel estimation.
- Done some tidying up in the code.

Thoughts:
- The BER is fixed at around 0.17 when SNR is 20dB. The way to improve this is to add correction code like LDPC, which is the next step.
