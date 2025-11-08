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
from preprocessing.members_matching import matching_members, fit_propensity_model, match_on_propensity, \
    validate_matching_quality
from utils.data_loaders import get_features_df, get_churn_labels_df
from sklift.models.models import TwoModels
from xgboost import XGBClassifier

import plotly.io as pio

from utils.explain_model import plot_logistic_regression_importance
from utils.plot_utils import plot_feature_correlation_heatmap

pio.renderers.default = "browser"

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from utils.metrics import c_for_benefit_from_pairs


TEST_MODE = os.getenv("TEST_MODE") == "1"
if TEST_MODE:
    pio.renderers.default = "json"


def safe_show(fig: go.Figure):
    if not TEST_MODE:
        fig.show()


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


def run_training():
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
    X = X[
        [
            'total_app_sessions',
            'total_wellco_web_visits',
            'unique_urls',
            'unique_wellco_web_active_days',
            'wellco_web_visits_ratio',
            'total_web_visits',
            'has_dietary',
            'app_usage_duration_days',
            'has_hypertension',
            'average_wellco_web_visits_per_active_day',
            'average_app_sessions_per_active_day',
            'days_from_last_app_use',
            'in_cohort',
            'has_diabetes',
            'max_app_sessions_per_day',
            'unique_app_active_days',
            'std_app_sessions_per_day',
        ]
    ]
    y = features_with_labels[CHURN_COLUMN]
    t = features_with_labels[OUTREACH_COLUMN]

    stratify_col = features_with_labels[[CHURN_COLUMN, OUTREACH_COLUMN]].astype(int).astype(str).agg('_'.join, axis=1)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t,
        test_size=0.2,
        stratify=stratify_col,
        random_state=42
    )

    X_train_m, t_train_m, y_train_m, matched_idx_train, pairs_train = matching_members(X_train, t_train, y_train)

    plot_feature_correlation_heatmap(X_train_m, "Correlation Heatmap - Matched Train Set")

    baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
    )

    baseline_pipeline = make_pipeline(
        StandardScaler(),
        baseline_model,
    )
    baseline_pipeline.fit(X_train_m, y_train_m)

    y_pred = baseline_pipeline.predict(X_test)
    y_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

    print("AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    print("ROC AUC on Train:", roc_auc_score(y_train_m, baseline_pipeline.predict_proba(X_train_m)[:, 1]))

    importance_df, fig = plot_logistic_regression_importance(
        model=baseline_pipeline,
        X_train=X_train_m,
        output_dir=os.path.join(ARTIFACTS_DIRECTORY_PATH, "explainability"),
        top_n=30,
        model_name="baseline_logistic_regression"
    )
    safe_show(fig)

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
        X_train_m[t_train_m == 1],
        y_train_m[t_train_m == 1],
    )

    control_pipeline.fit(
        X_train_m[t_train_m == 0],
        y_train_m[t_train_m == 0],
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

    ctrl_rate = ((t_ord == 0) & (y_ord == 1)).sum() / max((t_ord == 0).sum(), 1)
    qini = cum_y_t - ctrl_rate * cum_treat

    optimal_n = int(np.argmax(qini)) + 1
    print(f"Optimal n = {optimal_n}")

    ## Uplift with cost and value assumptions
    for v in [20]:
        for c in [1, 0.6, 0.06, 0.006]:
            uplift_sorted = np.sort(uplift_all)[::-1]
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
                title_x=0.5,
                xaxis_title="Number of Members Outreached (sorted by uplift)",
                yaxis_title="Cumulative Net Value (USD)",
                template="plotly_white",
            )
            safe_show(fig)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=uplift_all,
        nbinsx=50,
        marker=dict(
            color='rgba(0, 102, 204, 0.7)',
            line=dict(width=1, color='white')
        ),
        name='Predicted uplift',
        hovertemplate='Uplift: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))

    fig.add_vline(
        x=np.mean(uplift_all),
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean = {np.mean(uplift_all):.3f}",
        annotation_position="top right"
    )

    fig.add_vline(
        x=np.median(uplift_all),
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median = {np.median(uplift_all):.3f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title=dict(
            text="Distribution of Predicted Uplift (Control − Treatment)",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial Black")
        ),
        xaxis_title="Predicted Uplift",
        yaxis_title="Number of Members",
        bargap=0.05,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=False)

    # Sort uplift descending
    sorted_indices = np.argsort(-uplift_all)
    uplift_sorted = uplift_all[sorted_indices]
    cum_gain = np.cumsum(uplift_sorted)

    # Determine optimal n (reuse the one you already computed)
    optimal_n = int(np.argmax(cum_gain)) + 1

    fig = go.Figure()

    # Main cumulative uplift line
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(cum_gain) + 1),
        y=cum_gain,
        mode='lines',
        line=dict(width=2, color='rgba(0,102,204,0.9)'),
        name='Cumulative Uplift',
        hovertemplate='Members Reached: %{x}<br>Cumulative Uplift: %{y:.2f}<extra></extra>'
    ))

    # Optimal n vertical line
    fig.add_vline(
        x=optimal_n,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal n = {optimal_n}",
        annotation_position="top right"
    )

    # Layout styling
    fig.update_layout(
        title=dict(
            text="Cumulative Uplift Curve (Optimal n Highlighted)",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial Black")
        ),
        xaxis_title="Number of Members (sorted by uplift)",
        yaxis_title="Cumulative Expected Retention Gain",
        template="plotly_white",
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=False)

    safe_show(fig)

    best_net_value = net_curve[optimal_n - 1]
    roi = best_net_value / (c * optimal_n) if c * optimal_n > 0 else float('inf')
    print(f"[Cost={c}, Value={v}] ROI at optimal n: {roi:.2f}x | Expected net gain: ${best_net_value:.2f}")

    y_retention_test = 1 - y_test
    y_retention_train = 1 - y_train_m

    xgb_params = dict(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    baseline_xgb_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    treatment_pipeline = make_pipeline(
        StandardScaler(),
        XGBClassifier(**baseline_xgb_params)
    )

    control_pipeline = make_pipeline(
        StandardScaler(),
        XGBClassifier(**baseline_xgb_params)
    )

    uplift_model = TwoModels(
        estimator_trmnt=treatment_pipeline,
        estimator_ctrl=control_pipeline,
        method='vanilla'
    )

    uplift_model.fit(X_train_m, y_retention_train, treatment=t_train_m)
    uplift_predictions_test = uplift_model.predict(X_test)
    uplift_predictions_all = uplift_model.predict(X)
    uplift_predictions_train_m = uplift_model.predict(X_train_m)

    uplift_train_full = np.full(len(X_train), np.nan, dtype=float)
    uplift_train_full[matched_idx_train] = uplift_predictions_train_m

    sorted_indices = np.argsort(-uplift_all)
    uplift_sorted = uplift_all[sorted_indices]
    cum_gain = np.cumsum(uplift_sorted)

    optimal_n = int(np.argmax(cum_gain)) + 1
    budget_n = 3959
    optimal_n = budget_n
    print(f"Final optimal n = {optimal_n}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(cum_gain) + 1),
        y=cum_gain,
        mode='lines',
        line=dict(width=2, color='rgba(0,102,204,0.9)'),
        name='Cumulative Uplift',
        hovertemplate='Members Reached: %{x}<br>Cumulative Uplift: %{y:.2f}<extra></extra>'
    ))
    fig.add_vline(
        x=optimal_n,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal n = {optimal_n}",
        annotation_position="top right"
    )
    fig.update_layout(
        title=dict(
            text="Cumulative Uplift Curve (Optimal n Highlighted)",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial Black")
        ),
        xaxis_title="Number of Members (sorted by uplift)",
        yaxis_title="Cumulative Expected Retention Gain",
        template="plotly_white",
        showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=False)
    safe_show(fig)

    output_outreach_suggestion_path = os.path.join(
        output_predictions_dir_path,
        f'outreach_suggestion_{version}.csv'
    )
    outreach_suggestion_df = predicted_uplift_all.head(optimal_n)
    outreach_suggestion_df.to_csv(output_outreach_suggestion_path, index=False)

    ps_model, e_train = fit_propensity_model(X_train, t_train)
    e_test = ps_model.predict_proba(X_test)[:, 1]

    matched_idx_test, pairs_test = match_on_propensity(e=e_test, t=t_test, caliper='auto', replace=False)

    cfb_test = c_for_benefit_from_pairs(
        y=y_test.to_numpy(),
        uplift=-uplift_predictions_test,
        pairs=pairs_test
    )
    print(f"C-for-Benefit (test): {cfb_test:.3f} (pairs={len(pairs_test)})")

    cfb_train = c_for_benefit_from_pairs(
        y=y_train.to_numpy(),  # full train order
        uplift=-uplift_train_full,  # negate because y is churn, uplift is retention
        pairs=pairs_train
    )
    print(f"C-for-Benefit (train): {cfb_train:.3f} (pairs={len(pairs_train)})")

    actual_treated = t_test == 1
    actual_retention_rate = (1 - y_test[actual_treated]).mean()
    recommended_idx = np.argsort(uplift_predictions_test)[-optimal_n:]
    simulated_retention_rate = (1 - y_test.iloc[recommended_idx]).mean()
    delta_retention = simulated_retention_rate - actual_retention_rate
    uplift_percent = 100 * delta_retention / actual_retention_rate
    print(f"Retention gain = {delta_retention:.3f} ({uplift_percent:.2f}% improvement)")
    print(f"Predicted retention = {simulated_retention_rate: .3f} vs. Current retention = {actual_retention_rate: .3f}")

    roi_model = (v * delta_retention * len(y_test) - c * optimal_n) / (
            c * optimal_n)
    print(f"ROI vs historical: {roi_model:.1f}x")

    validate_matching_quality(X_train, t_train, X_train_m, t_train_m)

    treated_retention = (1 - y_train_m[t_train_m == 1]).mean()
    control_retention = (1 - y_train_m[t_train_m == 0]).mean()
    ate_historical = treated_retention - control_retention
    print(f"Current ATE (Average Treatment Effect) on matched data: {ate_historical:.3%}")

    uplift_df = pd.DataFrame({
        MEMBER_ID_COLUMN: features_with_labels[MEMBER_ID_COLUMN],
        "uplift": uplift_predictions_all
    }).sort_values(by="uplift", ascending=False)

    # Select the top 'optimal_n' members the model recommends for outreach
    treated_optimal = uplift_df.head(optimal_n)

    # Expected average uplift among those top-n members
    expected_uplift_new_policy = treated_optimal["uplift"].mean()
    print(f"Expected uplift (model-based policy): {expected_uplift_new_policy:.3%}")

    # Compare performance
    delta_vs_historical = expected_uplift_new_policy - ate_historical
    improvement_ratio = expected_uplift_new_policy / ate_historical if ate_historical != 0 else np.nan
    print(f"Improvement over historical: {delta_vs_historical:.3%} ({improvement_ratio:.2f}x)")


if __name__ == '__main__':
  run_training()
