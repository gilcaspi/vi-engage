import os

from data.features import FEATURES_DIRECTORY_PATH
from preprocessing.features_generation import generate_features
from preprocessing.web_visits_classification import add_semantic_web_category_features
from utils.data_loaders import get_web_visits_df

if __name__ == '__main__':
    features_version = 'v2'
    features_df = generate_features()

    features_df = add_semantic_web_category_features(
        web_visits_df=get_web_visits_df(),
        member_features=features_df,
        n_categories=10,
        random_state=23
    )

    output_file_path = os.path.join(FEATURES_DIRECTORY_PATH, f'generated_features_{features_version}.csv')
    features_df.to_csv(output_file_path, index=False)
