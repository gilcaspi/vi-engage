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

    churn_labels_df['signup_date'] = pd.to_datetime(churn_labels_df['signup_date'], format="%Y-%m-%d")
    churn_labels_df = churn_labels_df.sort_values(by='signup_date')
    return churn_labels_df


def get_app_usage_df() -> pd.DataFrame:
    app_usage_file_path = os.path.join(RAW_DATA_DIRECTORY_PATH, 'app_usage.csv')
    app_usage_df = pd.read_csv(app_usage_file_path)

    app_usage_df['timestamp'] = pd.to_datetime(app_usage_df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    app_usage_df = app_usage_df.sort_values(by='timestamp')
    return app_usage_df


def get_web_visits_df() -> pd.DataFrame:
    web_visits_file_path = os.path.join(RAW_DATA_DIRECTORY_PATH, 'web_visits.csv')
    web_visits_df = pd.read_csv(web_visits_file_path)

    web_visits_df['timestamp'] = pd.to_datetime(web_visits_df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    web_visits_df = web_visits_df.sort_values(by='timestamp')
    return web_visits_df
