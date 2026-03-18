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

## version 2.3
Thoughts:
- Adjusted pilot indices and use estimated noise in receiver (of course). Spend some time to make some other adjustments.
- BER rate is 0.17 at SNR=20db, which is acceptable just from this raw transmission. I think to implement error correction code later.
- But which to use? I'm thinking LDPC. Just because it's what's being used in real 5G cases. We'll see. 
- Hopefully BER can drop significantly and then i can probably change from 16 QAM to 256QAM (such a big leap).
- Oh, another thing is that the demodulation method needs a bit improvement. ML is quite simple.

## version 2.4
Thoughts:
- In this version, I'll try to implement LDPC code. If failed i will choose simpler error correction code.
- I'll change see if i can change the demodulation method a bit.
- If the BER rate is low enough, i will try to change to 256QAM.