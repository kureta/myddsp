"""Constants for sample pre-processing."""

SAMPLE_RATE = 48000
CREPE_SAMPLE_RATE = 16000
SR_RATIO = SAMPLE_RATE // CREPE_SAMPLE_RATE
CREPE_N_FFT = 1024
N_FFT = CREPE_N_FFT * SR_RATIO

# TODO: frame rate can be a parameter
#       only constraint is sample rate has to be divisible by frame rate
FRAME_RATE = 250
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
CREPE_HOP_LENGTH = HOP_LENGTH // SR_RATIO
CREPE_N_BINS = 360

DYNAMIC_RANGE = 70.0
