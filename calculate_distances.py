import MFCC
import DTW
import numpy as np
import sys
import os
import csv


def calculate_within_cluster_distance(filepath):
    feature_vectors = list()
    samples = os.listdir(filepath)
    for sample in samples:
        feature_vectors.append(MFCC.voice_feature_extraction(os.path.join(filepath, sample)))

    distances = list()
    for i in range(0, len(feature_vectors)):
        for j in range(i+1, len(feature_vectors)):
            distances.append(DTW.dynamic_time_warping(feature_vectors[i], feature_vectors[j]))

    return np.max(distances)


if len(sys.argv) != 2:
    print("Usage: calculate_distance <path_to_database>")
    exit(-1)

database_filepath = sys.argv[1]
persons = os.listdir(database_filepath)

with open('cluster_distances.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'distance'])
    for person in persons:
        words = os.listdir(os.path.join(database_filepath, person))
        for word in words:
            samples_path = os.path.join(database_filepath, person, word)
            print(samples_path)
            distance = calculate_within_cluster_distance(samples_path)
            writer.writerow([samples_path, distance])



