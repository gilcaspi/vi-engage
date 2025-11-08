import os
import numpy as np
import pandas as pd
import plotly.express as px


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
