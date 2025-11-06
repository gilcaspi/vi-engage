import os

import pandas as pd

from data.raw import RAW_DATA_DIRECTORY_PATH


def get_claims_df() -> pd.DataFrame:
    claims_file_path = os.path.join(RAW_DATA_DIRECTORY_PATH, 'claims.csv')
    claims_df = pd.read_csv(claims_file_path)
    return claims_df


def get_churn_labels_df() -> pd.DataFrame:
    churn_labels_file_path = os.path.join(RAW_DATA_DIRECTORY_PATH, 'churn_labels.csv')
    churn_labels_df = pd.read_csv(churn_labels_file_path)
    return churn_labels_df