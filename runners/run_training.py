import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from artifacts import ARTIFACTS_DIRECTORY_PATH
from data.raw.raw_data_column_names import MEMBER_ID_COLUMN, OUTREACH_COLUMN, CHURN_COLUMN, SIGNUP_DATE_COLUMN
from utils.data_loaders import get_features_df, get_churn_labels_df

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


if __name__ == '__main__':
    features_df = get_features_df(features_version='v1')
    churn_labels_df = get_churn_labels_df()

    features_with_labels = features_df.merge(
        churn_labels_df,
        on=MEMBER_ID_COLUMN,
        how='left'
    )

    X = features_with_labels.drop(
        columns=[
            MEMBER_ID_COLUMN,
            OUTREACH_COLUMN,
            CHURN_COLUMN,
            SIGNUP_DATE_COLUMN,
        ],
        errors="ignore",
    )
    y = features_with_labels[CHURN_COLUMN]
    t = features_with_labels[OUTREACH_COLUMN]

    stratify_col = features_with_labels[[CHURN_COLUMN, OUTREACH_COLUMN]].astype(int).astype(str).agg('_'.join, axis=1)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t,
        test_size=0.2,
        stratify=stratify_col,
        random_state=42
    )

    baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
    )

    baseline_pipeline = make_pipeline(
        StandardScaler(),
        baseline_model,
    )
    baseline_pipeline.fit(X_train, y_train)

    y_pred = baseline_pipeline.predict(X_test)
    y_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

    print("AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    version = 'v1'
    model_name = f'baseline_logistic_regression_model'

    output_models_dir_path = os.path.join(ARTIFACTS_DIRECTORY_PATH, 'models')
    os.makedirs(output_models_dir_path, exist_ok=True)

    output_model_baseline_path = os.path.join(output_models_dir_path, f'{model_name}_{version}.pkl')
    joblib.dump(baseline_pipeline, output_model_baseline_path)

    predicted_churn_probabilities = pd.DataFrame({
        MEMBER_ID_COLUMN: features_with_labels[MEMBER_ID_COLUMN],
        "churn_prob": baseline_pipeline.predict_proba(X)[:, 1]
    })
    predicted_churn_probabilities.sort_values(by='churn_prob', ascending=False, inplace=True)

    output_predictions_dir_path = os.path.join(ARTIFACTS_DIRECTORY_PATH, 'predictions')
    os.makedirs(output_predictions_dir_path, exist_ok=True)

    output_predictions_path = os.path.join(
        output_predictions_dir_path,
        f'{model_name}_probabilities_{version}.csv'
    )
    predicted_churn_probabilities.to_csv(output_predictions_path, index=False)


    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], '--', color='orange', label='Perfect calibration')
    plt.legend()
    plt.title("Calibration Curve", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Observed Churn Rate", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()