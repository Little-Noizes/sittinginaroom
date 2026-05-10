PROJECT GOAL
Investigate recursive electroacoustic playback as a method for
revealing dominant resonant behaviour of a measured room transfer
function.

MATHEMATICAL MODEL
x_{n+1}(t) = G [x_n(t) * h(t)]

Two processing paths:
1. Physical fixed-gain recursive playback
2. Normalised recursive modal extraction

INPUTS
- mono wav input signal
- measured room impulse response wav
- sample rate
- loop gain
- number of generations

OUTPUTS
- wav per generation
- FFT plots
- spectrograms
- modal peak tracking CSV
- theoretical modal frequency table

NORMALISATION OPTIONS
- none
- peak
- RMS
- low-frequency RMS (20-200 Hz)

ANALYSIS
- Long FFT windows
- logarithmic frequency axis
- compare against theoretical rectangular room modes


IMPORTANT SCIENTIFIC CONSTRAINT
The recursive process does not directly isolate room modes.
It progressively amplifies dominant poles of the coupled
source-room-receiver transfer function.

PREFERRED STACK
- Python
- numpy
- scipy
- librosa
- matplotlib
- soundfile

DIRECTORY STRUCTURE
...