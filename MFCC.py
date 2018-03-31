import numpy as np
import scipy.io.wavfile
import scipy.fftpack


def pre_emphasis(x, a=0.97):
    """
    Enhances higher frequencies in the signal
    :param x: input signal
    :param a: pre-emphasis coefficient
    :return: signal with enhanced higher frequencies
    """
    x[1:] = x[1:] - a * x[0:len(x)-2]
    return x


def voice_feature_extraction(infile):
    """
    Calculates the MFCC vectors of a given signal
    :param infile: audio file (.wav format)
    :return: vector of MFCC features
    """

    # read audio file
    sampling_rate, signal = scipy.io.wavfile.read(infile)

    # divide into n[ms] overlapping frames
    n = 30
    samples_per_frame = int(sampling_rate/1000 * n)
    frame_step = int(samples_per_frame/2)
    frame_count = int(len(signal) / frame_step)

    frames = list()
    for i in range(0, frame_count):
        frames.append(signal[(i*frame_step):(i*frame_step + samples_per_frame)])

    # calculate MFCC vector for each frame
    feature_vector = list()
    for frame in frames:
        feature_vector.append(MFCC(frame * np.hamming(len(frame)), sampling_rate))

    return feature_vector


def MFCC(windowed_frame, sampling_rate):
    """
    Calculates the MFCC vector of single frame
    :param windowed_frame: windowed frame of audio input
    :return: MFCC vector
    """

    # calculate power spectrum
    spectrum = np.fft.fft(windowed_frame)
    power_spectrum = np.square(np.abs(spectrum))

    # generate triangular filters
    triangle_filters = np.ones((12, len(power_spectrum)))

    # Apply filters to power spectrum
    S = triangle_filters * np.tile(power_spectrum, (12, 1))

    return scipy.fftpack.dct(np.log10(S), norm='ortho')[:, 1:13]
