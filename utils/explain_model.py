import os
from typing import Union

import numpy as np
import shap
from matplotlib import pyplot as plt
from numpy.typing import NDArray
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline


def plot_logistic_regression_importance(
        model,
        X_train: pd.DataFrame,
        output_dir: str,
        top_n: int = 30,
        model_name: str = "logistic_regression"
):
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        log_reg = model.named_steps["logisticregression"]
    else:
        log_reg = model

    if not hasattr(log_reg, "coef_"):
        raise ValueError("Model does not have coefficients (did you fit it?)")

    feature_names = X_train.columns
    coefs = log_reg.coef_.flatten()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs)
    }).sort_values("Abs_Coefficient", ascending=False)

    fig = px.bar(
        importance_df.head(top_n),
        x="Abs_Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu",
        title=f"{model_name}: Top {top_n} Standardized Coefficients",
        labels={"Abs_Coefficient": "Absolute Coefficient (Importance)", "Feature": "Feature"},
        height=800
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        title_x=0.5,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(output_path, index=False)

    fig.write_html(os.path.join(output_dir, f"{model_name}_feature_importance.html"))
    print(f"Saved logistic regression feature importances to: {output_path}")

    return importance_df, fig


def compute_uplift_shap(
        model_control_pipeline: Pipeline,
        model_treatment_pipeline: Pipeline,
        X: Union[pd.DataFrame, NDArray],
        plot: bool = True,
        max_samples: int = 1000,
) -> NDArray:
    """
    Compute SHAP values for uplift from a Two-Model XGB setup,
    where each model is an sklearn Pipeline: [preprocessor] -> [xgb_model].

    uplift(x) = p0(x) - p1(x)
    """

    if hasattr(X, "sample"):
        X_sample = X.sample(min(max_samples, len(X)), random_state=42)
    else:
        idx = np.random.choice(len(X), size=min(max_samples, len(X)), replace=False)
        X_sample = X[idx]

    feature_names = list(X_sample.columns) if hasattr(X_sample, "columns") else None

    preproc_ctrl = model_control_pipeline.named_steps["preprocessor"]
    xgb_ctrl = model_control_pipeline.named_steps["model"]

    preproc_trmt = model_treatment_pipeline.named_steps["preprocessor"]
    xgb_trmt = model_treatment_pipeline.named_steps["model"]

    X_ctrl_trans = preproc_ctrl.transform(X_sample)
    X_trmt_trans = preproc_trmt.transform(X_sample)

    if hasattr(X_ctrl_trans, "toarray"):
        X_ctrl_trans = X_ctrl_trans.toarray()
    if hasattr(X_trmt_trans, "toarray"):
        X_trmt_trans = X_trmt_trans.toarray()

    def predict_ctrl(data: NDArray) -> NDArray:
        return xgb_ctrl.predict_proba(data)[:, 1]

    def predict_trmt(data: NDArray) -> NDArray:
        return xgb_trmt.predict_proba(data)[:, 1]

    expl_ctrl = shap.Explainer(predict_ctrl, X_ctrl_trans)
    shap_ctrl = expl_ctrl(X_ctrl_trans).values

    expl_trmt = shap.Explainer(predict_trmt, X_trmt_trans)
    shap_trmt = expl_trmt(X_trmt_trans).values

    shap_uplift = shap_ctrl - shap_trmt

    if plot:
        if feature_names is not None and shap_uplift.shape[1] == len(feature_names):
            expl_uplift = shap.Explanation(
                values=shap_uplift,
                data=X_ctrl_trans,
                feature_names=feature_names,
            )
        else:
            expl_uplift = shap.Explanation(
                values=shap_uplift,
                data=X_ctrl_trans,
            )

        # 1) Global uplift importance – bar plot
        plt.figure(figsize=(50, 20))
        shap.plots.bar(expl_uplift, max_display=30, show=False)
        fig = plt.gcf()
        fig.subplots_adjust(left=0.35)  # more space for long feature names
        plt.tight_layout()
        plt.show()

        # 2) Uplift summary / beeswarm plot
        plt.figure(figsize=(50, 20))
        shap.summary_plot(
            shap_uplift,
            X_ctrl_trans,
            feature_names=feature_names,
            max_display=20,
            show=False,
            plot_size=(15, 10),
        )
        fig = plt.gcf()
        fig.subplots_adjust(left=0.35)
        plt.tight_layout()
        plt.show()

    return shap_uplift
