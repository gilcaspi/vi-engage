import pandas as pd

from utils.data_loaders import get_app_usage_df, get_web_visits_df, get_claims_df


def generate_features() -> pd.DataFrame:
    claims_df = get_claims_df()

    icd_code_strip = claims_df['icd_code'].str.strip()

    claims_df['has_diabetes'] = icd_code_strip == 'E11.9'
    claims_df['has_hypertension'] = icd_code_strip == 'I10'
    claims_df['has_dietary'] = icd_code_strip == 'Z71.3'

    member_features = claims_df.groupby('member_id')[
        ['has_diabetes', 'has_hypertension', 'has_dietary']
    ].any().reset_index()

    member_features['in_cohort'] = member_features[
        ['has_diabetes', 'has_hypertension', 'has_dietary']
    ].any(axis=1)

    app_usage_df = get_app_usage_df()
    app_usage_df['date'] = app_usage_df['timestamp'].dt.date

    member_features['total_app_sessions'] = app_usage_df.groupby('member_id').size()
    member_features['unique_app_active_days'] = app_usage_df.groupby('member_id')['date'].nunique()
    member_features['average_app_sessions_per_active_day'] = (
            member_features['total_app_sessions'] / member_features['unique_app_active_days']
    )

    daily_app_sessions = app_usage_df.groupby(['member_id', 'date']).size().reset_index(name='daily_sessions')
    member_features['max_app_sessions_per_day'] = daily_app_sessions.groupby('member_id')['daily_sessions'].max()
    member_features['std_app_sessions_per_day'] = daily_app_sessions.groupby('member_id')['daily_sessions'].std()

    first_app_session_date = app_usage_df.groupby('member_id')['date'].min()
    last_app_session_date = app_usage_df.groupby('member_id')['date'].max()

    member_features['app_usage_duration_days'] = (
            last_app_session_date - first_app_session_date
    )

    member_features['days_from_last_app_use'] = (
        pd.to_datetime('today').date() - last_app_session_date
    ).dt.days

    web_visits_df = get_web_visits_df()

    member_features['total_web_visits'] = web_visits_df.groupby('member_id').size()

    wellco_visits_df = web_visits_df['url'].str.contains('wellco', case=False, na=False)


    return member_features


if __name__ == '__main__':
    features = generate_features()
    print()
