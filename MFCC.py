import numpy as np


def pre_emphasis(x, a=0.97):
    """
    Enhances higher frequencies in the signal
    :param x: input signal
    :param a: pre-emphasis coefficient
    :return: signal with enhanced higher frequencies
    """
    x[1:] = x[1:] - a * x[0:len(x)-2]
    return x

def voice_feature_extraction(input_signal):


def MFCC(windowed_frame):
    """
    Calculates the MFCC vector of single frame
    :param windowed_frame: windowed frame of audio input
    :return: MFCC vector
    """

    X = np.fft(windowed_frame)
    np.square(np.abs(X))

def hamming_window(frame_length, frame_start):
    t = frame_start + np.array(range(0, frame_length))
    return 0.54 - 0.46*np.cos(t)



