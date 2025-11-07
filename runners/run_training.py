import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from artifacts import ARTIFACTS_DIRECTORY_PATH
from data.raw.raw_data_column_names import MEMBER_ID_COLUMN, OUTREACH_COLUMN, CHURN_COLUMN, SIGNUP_DATE_COLUMN
from utils.data_loaders import get_features_df, get_churn_labels_df

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def plot_calibration_curve(
        y_ground_truth: pd.Series,
        y_estimated_probability: pd.Series,
        model_name: str='Model'
) -> None:
    prob_true, prob_pred = calibration_curve(y_ground_truth, y_estimated_probability, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], '--', color='orange', label='Perfect calibration')
    plt.legend()
    plt.title("Calibration Curve", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Observed Churn Rate", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


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
    baseline_model_name = f'baseline_logistic_regression_model'

    output_models_dir_path = os.path.join(ARTIFACTS_DIRECTORY_PATH, 'models')
    os.makedirs(output_models_dir_path, exist_ok=True)

    output_model_baseline_path = os.path.join(output_models_dir_path, f'{baseline_model_name}_{version}.pkl')
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
        f'{baseline_model_name}_probabilities_{version}.csv'
    )
    predicted_churn_probabilities.to_csv(output_predictions_path, index=False)

    plot_calibration_curve(
        y_ground_truth=y_test,
        y_estimated_probability=y_proba,
        model_name=baseline_model_name,
    )

    ## Uplift
    treatment_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
        ),
    )

    control_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
        ),
    )

    treatment_pipeline.fit(
        X_train[t_train == 1],
        y_train[t_train == 1],
    )

    control_pipeline.fit(
        X_train[t_train == 0],
        y_train[t_train == 0],
    )

    treatment_proba = treatment_pipeline.predict_proba(X_test)[:, 1]
    control_proba = control_pipeline.predict_proba(X_test)[:, 1]

    uplift = control_proba - treatment_proba
    uplift_model_name = f'uplift_logistic_regression_model'
    output_model_uplift_path = os.path.join(output_models_dir_path, f'{uplift_model_name}_{version}.pkl')
    joblib.dump({
        "treatment_model": treatment_pipeline,
        "control_model": control_pipeline,
    }, output_model_uplift_path)

    predicted_uplift = pd.DataFrame({
        MEMBER_ID_COLUMN: features_with_labels.loc[X_test.index, MEMBER_ID_COLUMN].values,
        "uplift": uplift
    })
    predicted_uplift.sort_values(by='uplift', ascending=False, inplace=True)

    output_uplift_predictions_path = os.path.join(
        output_predictions_dir_path,
        f'{uplift_model_name}_predictions_{version}.csv'
    )
    predicted_uplift.to_csv(output_uplift_predictions_path, index=False)


    ## Uplift all members
    treatment_proba_all = treatment_pipeline.predict_proba(X)[:, 1]
    control_proba_all = control_pipeline.predict_proba(X)[:, 1]
    uplift_all = control_proba_all - treatment_proba_all
    predicted_uplift_all = pd.DataFrame({
        MEMBER_ID_COLUMN: features_with_labels[MEMBER_ID_COLUMN],
        "uplift": uplift_all
    })
    predicted_uplift_all.sort_values(by='uplift', ascending=False, inplace=True)
    predicted_uplift_all = predicted_uplift_all.copy()
    predicted_uplift_all['rank'] = np.arange(1, len(predicted_uplift_all) + 1)
    predicted_uplift_all.rename(columns={'uplift': 'prioritization_score'}, inplace=True)

    output_uplift_all_predictions_path = os.path.join(
        output_predictions_dir_path,
        f'{uplift_model_name}_predictions_all_{version}.csv'
    )
    predicted_uplift_all.to_csv(output_uplift_all_predictions_path, index=False)

    order = np.argsort(-uplift)
    y_ord = y_test.values[order]
    t_ord = t_test.values[order]

    cum_treat = (t_ord == 1).astype(int).cumsum()
    cum_ctrl = (t_ord == 0).astype(int).cumsum()
    cum_y_t = ((t_ord == 1) & (y_ord == 1)).astype(int).cumsum()

    ctrl_rate = ((t_ord == 0) & (y_ord == 1)).sum() / max((t_ord == 0).sum(), 1)  # baseline churn in control
    qini = cum_y_t - ctrl_rate * cum_treat

    optimal_n = int(np.argmax(qini)) + 1
    print(f"Optimal n = {optimal_n}")

    ## Uplift with cost and value assumptions
    for v in [20]:
        for c in [0.6, 0.06, 0.006]:
            uplift_sorted = np.sort(uplift)[::-1]
            net_curve = np.cumsum(v * uplift_sorted - c)
            optimal_n = int(np.argmax(net_curve)) + 1
            print(f"Optimal n considering cost = {optimal_n}")


            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(net_curve) + 1),
                y=net_curve,
                mode='lines',
                name='Cumulative Net Value',
                line=dict(width=2)
            ))
            fig.add_vline(
                x=optimal_n,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Optimal n = {optimal_n}",
                annotation_position="top right"
            )
            fig.update_layout(
                title=f"Net Value vs. n (Cost = {c}, Value per uplift unit = {v})",
                xaxis_title="Number of Members Outreached (sorted by uplift)",
                yaxis_title="Cumulative Net Value (USD)",
                template="plotly_white",
            )
            fig.show()