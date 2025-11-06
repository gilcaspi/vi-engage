# Schema: churn_labels.csv

- member_id (int): Unique member identifier.
- signup_date (date): Member signup date (YYYY-MM-DD).
- churn (int, {0,1}): Target label indicating near-term churn after the observation window.
- outreach (int, {0,1}): Binary flag indicating whether the member received outreach (treatment).