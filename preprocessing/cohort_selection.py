import pandas as pd

from utils.data_loaders import get_claims_df


def get_cohort() -> pd.Series:
    full_members_df = get_claims_df()

    icd10_in_cohort = ['E11.9', 'I10', 'Z71.3']

    in_cohort_mask = full_members_df['icd_code'].isin(icd10_in_cohort)
    cohort_df = full_members_df[in_cohort_mask]

    cohort_member_ids = cohort_df['member_id']
    cohort_member_ids_no_duplication = cohort_member_ids.drop_duplicates().reset_index(drop=True)

    return cohort_member_ids_no_duplication


def get_control_group() -> pd.Series:
    full_members_df = get_claims_df()
    cohort_member_ids = get_cohort()

    not_in_cohort_mask = ~full_members_df['member_id'].isin(cohort_member_ids)
    control_group_df = full_members_df[not_in_cohort_mask]

    control_member_ids = control_group_df['member_id']
    control_member_ids_no_duplications = control_member_ids.drop_duplicates().reset_index(drop=True)

    return control_member_ids_no_duplications


def get_full_member_ids() -> pd.Series:
    full_members_df = get_claims_df()
    full_member_ids = full_members_df['member_id']
    full_member_ids_no_duplication = full_member_ids.drop_duplicates().reset_index(drop=True)
    return full_member_ids_no_duplication
