import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot as plt


def pre_emphasis(x, a=0.97):
    """
    Enhances higher frequencies in the signal
    :param x: input signal
    :param a: pre-emphasis coefficient
    :return: signal with enhanced higher frequencies
    """
    x = x[1:] - a * x[:-1]
    return x


def voice_feature_extraction(infile):
    """
    Calculates the MFCC vectors of a given signal
    :param infile: audio file (.wav format)
    :return: vector of MFCC features
    """

    # read audio file
    sampling_rate, signal = scipy.io.wavfile.read(infile)

    # signal pre_emphasis
    signal = pre_emphasis(signal)

    # divide into n[ms] overlapping frames
    n = 25
    step_ms = 10
    samples_per_frame = int(sampling_rate/1000 * n)
    frame_step = int(sampling_rate/1000 * step_ms)
    frame_count = int(len(signal) / frame_step)

    frames = list()
    for i in range(0, frame_count):
        frames.append(signal[(i*frame_step):(i*frame_step + samples_per_frame)])

    # calculate MFCC vector for each frame
    feature_vector = list()
    for frame in frames:
        mfcc = MFCC(frame * np.hamming(len(frame)), sampling_rate)
        delta = derivative(mfcc)
        delta_delta = derivative(delta)
        mfcc = np.append(mfcc, delta)
        mfcc = np.append(mfcc, delta_delta)
        feature_vector.append(mfcc)

    return feature_vector


def MFCC(windowed_frame, sampling_rate):
    """
    Calculates the MFCC vector of single frame
    :param windowed_frame: windowed frame of audio input
    :return: MFCC vector
    """

    # calculate power spectrum
    spectrum = np.fft.fft(windowed_frame)
    power_spectrum = np.square(np.abs(spectrum))/len(spectrum)

    # generate triangular filters
    triangle_filters = get_filter_bank(sampling_rate, len(spectrum))

    # Apply filters to power spectrum
    S = triangle_filters.dot(power_spectrum)
    S = np.where(S == 0, np.finfo(float).eps, S)

    return lift(scipy.fftpack.dct(np.log(S), type=3, norm='ortho'))[1:12]


def get_filter_bank(sampling_rate, signal_length, count=24, lower_bound=0, upper_bound=800):
    lower_mel = frequency_to_mel(lower_bound)
    upper_mel = frequency_to_mel(sampling_rate/2)

    mel_samples = np.linspace(lower_mel, upper_mel, count + 2)
    frequency_samples = mel_to_frequency(mel_samples)

    frequency_bins = np.floor((signal_length+1) * frequency_samples / sampling_rate)

    filter_bank = np.zeros([count, signal_length])
    for j in range(0, count):
        for i in range(int(frequency_bins[j]), int(frequency_bins[j + 1])):
            filter_bank[j, i] = (i - frequency_bins[j]) / (frequency_bins[j + 1] - frequency_bins[j])
        for i in range(int(frequency_bins[j + 1]), int(frequency_bins[j + 2])):
            filter_bank[j, i] = (frequency_bins[j + 2] - i) / (frequency_bins[j + 2] - frequency_bins[j + 1])

    #for j in range(0, count):
    #   plt.plot(range(0,signal_length), filter_bank[j,:])

    #plt.show()

    return filter_bank


def derivative(mfcc_vector, N=2):
     return mfcc_vector[N:] - mfcc_vector[:-N]


def lift(mfcc_vector, L=22):
    n = np.arange(len(mfcc_vector))
    return mfcc_vector * (1 + L/2 * np.sin(np.pi * n /L))



def frequency_to_mel(f):
    """
    Converts frequency value(Hz) to mels
    :param f: frequency value(Hz)
    :return:
    """
    return 2595 * np.log10(1 + f/700.0)


def mel_to_frequency(mel):
    """
    Converts mels to frequency value(Hz)
    :param mel: mel value
    :return:
    """
    return 700 * (10**(mel/2595.0) - 1)

