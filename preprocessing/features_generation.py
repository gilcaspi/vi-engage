import numpy as np
import pandas as pd

from data.raw.raw_data_column_names import ICD10_CODE_COLUMN, MEMBER_ID_COLUMN
from utils.data_loaders import get_app_usage_df, get_web_visits_df, get_claims_df


def generate_features() -> pd.DataFrame:
    claims_df = get_claims_df()
    app_usage_df = get_app_usage_df()
    web_visits_df = get_web_visits_df()

    claims_df[MEMBER_ID_COLUMN] = claims_df[MEMBER_ID_COLUMN].astype(int)
    app_usage_df[MEMBER_ID_COLUMN] = app_usage_df[MEMBER_ID_COLUMN].astype(int)
    web_visits_df[MEMBER_ID_COLUMN] = web_visits_df[MEMBER_ID_COLUMN].astype(int)

    overlap_member_ids = (
        pd.Index(claims_df[MEMBER_ID_COLUMN].unique())
        .intersection(pd.Index(app_usage_df[MEMBER_ID_COLUMN].unique()))
        .intersection(pd.Index(web_visits_df[MEMBER_ID_COLUMN].unique()))
    )

    icd_code_strip = claims_df[ICD10_CODE_COLUMN].str.strip()

    claims_df['has_diabetes'] = icd_code_strip.eq('E11.9')
    claims_df['has_hypertension'] = icd_code_strip.eq('I10')
    claims_df['has_dietary'] = icd_code_strip.eq('Z71.3')

    # member_features = claims_df[[MEMBER_ID_COLUMN]].drop_duplicates().reset_index(drop=True)
    member_features = pd.DataFrame({MEMBER_ID_COLUMN: overlap_member_ids})

    claims_agg = (
        claims_df.groupby(MEMBER_ID_COLUMN)[["has_diabetes", "has_hypertension", "has_dietary"]]
        .any()
        .reset_index()
    )
    member_features = member_features.merge(claims_agg, on=MEMBER_ID_COLUMN, how="left")

    member_features['in_cohort'] = member_features[
        ['has_diabetes', 'has_hypertension', 'has_dietary']
    ].any(axis=1)

    app_usage_df['date'] = app_usage_df['timestamp'].dt.date

    total_app_sessions = (
        app_usage_df
        .groupby(MEMBER_ID_COLUMN)
        .size()
        .reindex(overlap_member_ids, fill_value=0)
        .rename("total_app_sessions")
        .rename_axis(MEMBER_ID_COLUMN)
        .reset_index()
    )

    unique_app_active_days = (
        app_usage_df.groupby(MEMBER_ID_COLUMN)['date'].nunique().rename("unique_app_active_days").reset_index()
    )

    daily_app_sessions = app_usage_df.groupby([MEMBER_ID_COLUMN, 'date']).size().reset_index(name='daily_app_sessions')

    max_app_sessions_per_day = (
        daily_app_sessions.groupby(MEMBER_ID_COLUMN)['daily_app_sessions']
        .max()
        .rename('max_app_sessions_per_day')
        .reset_index()
    )

    std_app_sessions_per_day = (
        daily_app_sessions.groupby(MEMBER_ID_COLUMN)['daily_app_sessions']
        .std()
        .fillna(0)
        .rename('std_app_sessions_per_day')
        .reset_index()
    )

    first_app_session_date = app_usage_df.groupby(MEMBER_ID_COLUMN)['date'].min()
    last_app_session_date = app_usage_df.groupby(MEMBER_ID_COLUMN)['date'].max()
    last_app_session_dataset_date_data = app_usage_df['date'].max()

    app_usage_duration_days = (
            pd.to_timedelta(last_app_session_date - first_app_session_date)
    ).dt.days.rename("app_usage_duration_days").reset_index()

    days_from_last_app_use = (
        pd.to_timedelta(last_app_session_dataset_date_data - last_app_session_date)
    ).dt.days.rename("days_from_last_app_use").reset_index()

    web_visits_df['date'] = web_visits_df['timestamp'].dt.date

    total_web_visits = (
        web_visits_df.groupby(MEMBER_ID_COLUMN).size()
        .reindex(overlap_member_ids, fill_value=0)
        .rename('total_web_visits')
        .rename_axis(MEMBER_ID_COLUMN).reset_index()
    )

    wellco_visits_mask = web_visits_df['url'].str.contains('wellco', case=False, na=False)
    wellco_visits_df = web_visits_df[wellco_visits_mask]

    total_wellco_web_visits = (
        wellco_visits_df.groupby(MEMBER_ID_COLUMN).size()
        .reindex(overlap_member_ids, fill_value=0)
        .rename('total_wellco_web_visits')
        .rename_axis(MEMBER_ID_COLUMN).reset_index()
    )

    unique_urls = (
        web_visits_df.groupby(MEMBER_ID_COLUMN)['url'].nunique()
        .reindex(overlap_member_ids, fill_value=0)
        .rename('unique_urls')
        .rename_axis(MEMBER_ID_COLUMN).reset_index()
    )

    unique_wellco_web_active_days = (
        wellco_visits_df.groupby(MEMBER_ID_COLUMN)['date'].nunique()
        .reindex(overlap_member_ids, fill_value=0)
        .rename('unique_wellco_web_active_days')
        .rename_axis(MEMBER_ID_COLUMN).reset_index()
    )

    for part in [
        total_app_sessions,
        unique_app_active_days,
        max_app_sessions_per_day,
        std_app_sessions_per_day,
        app_usage_duration_days,
        days_from_last_app_use,
        total_web_visits,
        total_wellco_web_visits,
        unique_urls,
        unique_wellco_web_active_days,
    ]:
        member_features = member_features.merge(part, on=MEMBER_ID_COLUMN, how='left')


    member_features = member_features[member_features[MEMBER_ID_COLUMN].isin(overlap_member_ids)]

    member_features['wellco_web_visits_ratio'] = (
            member_features['total_wellco_web_visits'].fillna(0)
            / member_features['total_web_visits'].replace(0, np.nan)
    ).fillna(0)

    member_features['average_app_sessions_per_active_day'] = (
            member_features['total_app_sessions'] / member_features['unique_app_active_days'].replace(0, np.nan)
    ).fillna(0)

    member_features['average_wellco_web_visits_per_active_day'] = (
        member_features['total_wellco_web_visits'] / member_features['unique_wellco_web_active_days'].replace(0, np.nan)
    ).fillna(0)

    member_features[MEMBER_ID_COLUMN] = member_features[MEMBER_ID_COLUMN].astype(int)
    member_features.sort_values(by=MEMBER_ID_COLUMN, inplace=True)
    return member_features
