# -*- coding: utf-8 -*-
import click
import logging
import wave
import struct
import os
import json
import numpy as np

logger = logging.getLogger()
logger.setLevel("INFO")


def extra_features(samples_directory, output_filepath):
    """
    Convers the recording database into a csv file.
    :param samples_directory: path to the recording database
    :param output_filepath: path where to save the csv file
    """
    label_list = os.listdir(samples_directory)
    json_data = list()

    for label in label_list:
        label_path = os.path.join(samples_directory, label)
        for word in os.listdir(label_path):
            word_path = os.path.join(label_path, word)
            for sample in os.listdir(word_path):
                sample_path = os.path.join(word_path, sample)
                logging.info("Processing {}...".format(sample_path))
                wave_file = wave.open(sample_path, 'r')
                signal = list()
                signal_length = wave_file.getnframes()
                for k in range(0, signal_length):
                    wave_data = wave_file.readframes(1)
                    data = struct.unpack("<h", wave_data)
                    signal.append(int(data[0]))
                json_data.append(extract_feature_vector(np.array(signal),
                                                    wave_file.getframerate()))
    file = open(output_filepath, 'w')
    file.write(json.dumps(json_data))
    file.close()


def extract_feature_vector(signal, sampling_rate):
    """
    :param signal: input signal - frames of the recording (type: np.array)
    """

    # signal pre_emphasis
    a = 0.97
    signal[1:] - a * signal[:-1]

    frames = divide_into_frames(signal, 30, 25, sampling_rate)
    mfcc_coefficients = [extract_mfcc(frame, sampling_rate).tolist() for frame in frames]

    return(mfcc_coefficients)


def divide_into_frames(signal,
                       frame_length,
                       frame_step,
                       sampling_rate):
    """
    Divides the signal into frames of length: frame_length[ms].
    The distances between frame beginnings is equal to frame_step.

    :param signal: input signal - frames of the recording (type: np.array)
    :frame_length: length of a single frame (type: int)
    :frame_step: distance between frame beginnings
    :sampling_rate: sampling_rate of the recording
    """

    frame_step = int(sampling_rate/1000 * frame_length)
    frame_count = int(np.floor(len(signal) / frame_step))

    return [signal[(i*frame_step):(i*frame_step + frame_step)]
            for i in range(0, frame_count)]


def extract_mfcc(frame, sampling_rate):
    """
    :frame: audio frame of the recording
    :sampling_rate: sampling_rate of the recording
    """

    # frame windowing
    windowed_frame = frame * np.hamming(len(frame))

    # calculate power spectrum
    spectrum = np.fft.fft(windowed_frame)
    power_spectrum = np.square(np.abs(spectrum))/len(spectrum)

    # generate triangular filters
    triangle_filters = get_filter_bank(sampling_rate, len(spectrum))

    # Apply filters to power spectrum
    S = triangle_filters.dot(power_spectrum)
    S = np.where(S == 0, np.finfo(float).eps, S)

    return dct(np.log(S))[1:12]


def get_filter_bank(sampling_rate,
                    signal_length,
                    count=24,
                    lower_bound=0,
                    upper_bound=800):
    lower_mel = frequency_to_mel(lower_bound)
    upper_mel = frequency_to_mel(sampling_rate/2)
    """
    Creates a bank of triangular filters.
    :param sampling_rate:
    """

    mel_samples = np.linspace(lower_mel, upper_mel, count + 2)
    frequency_samples = mel_to_frequency(mel_samples)

    frequency_bins = np.floor((signal_length+1) * frequency_samples / sampling_rate)

    filter_bank = np.zeros([count, signal_length])
    for j in range(0, count):
        for i in range(int(frequency_bins[j]), int(frequency_bins[j + 1])):
            filter_bank[j, i] = (i - frequency_bins[j]) / (frequency_bins[j + 1] - frequency_bins[j])
        for i in range(int(frequency_bins[j + 1]), int(frequency_bins[j + 2])):
            filter_bank[j, i] = (frequency_bins[j + 2] - i) / (frequency_bins[j + 2] - frequency_bins[j + 1])

    return filter_bank


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


def dct(y):
    """
    Performs the discrete cosine transform
    :param y: input array
    """
    N = len(y)
    y2 = np.empty(2*N, float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = np.fft.rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    extra_features(input_filepath, output_filepath)


if __name__ == '__main__':
    main()
