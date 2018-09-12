import MFCC
import DTW
import os
import numpy as np
import pandas
import sys


def calculate_cluster_distance(sample, database):
    distances = list()
    sample_feature_vector = MFCC.voice_feature_extraction(sample)
    for database_sample in os.listdir(database):
        database_feature_vector = MFCC.voice_feature_extraction(os.path.join(database, database_sample))
        distances.append(DTW.dynamic_time_warping(sample_feature_vector, database_feature_vector))

    return np.mean(distances)


def verify_voice_sample(sample, database):
    distance = calculate_cluster_distance(sample, database)
    df = pandas.read_csv('cluster_distances.csv')
    series = df.loc[df['path'] == database]['distance']
    print(distance)
    print(series.iloc[0])
    return distance <= series.iloc[0]


if __name__ == "__main__":
    result = verify_voice_sample(sys.argv[1], sys.argv[2])
    print('Accepted' if result else 'Denied')