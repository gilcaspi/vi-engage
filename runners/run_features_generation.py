import os

from data.features import FEATURES_DIRECTORY_PATH
from preprocessing.features_generation import generate_features

if __name__ == '__main__':
    features_version = 'v1'
    features_df = generate_features()

    output_file_path = os.path.join(FEATURES_DIRECTORY_PATH, f'generated_features_{features_version}.csv')
    features_df.to_csv(output_file_path, index=False)
