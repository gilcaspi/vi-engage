from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklift.metrics import uplift_at_k
from tqdm import tqdm


def pie_plot(
        labels: List[str],
        values: List[int],
        title: str,
        pull: Optional[List[float]] = None,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
                pull=pull,
                marker=dict(line=dict(color="#000000", width=1)),
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        showlegend=False,
    )
    return fig


def scatter_plot(
        df: pd.DataFrame,
        x_column_name: Optional[str] = None,
        y_column_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        mode: str = "markers+lines"
) -> go.Figure:
    fig = go.Figure()

    if x_column_name is None:
        x_values = df.index
        x_label = "index"
    else:
        x_values = df[x_column_name]
        x_label = x_column_name

    if y_column_names is None:
        y_column_names = [col for col in df.columns if col != x_column_name]

    if title is None:
        title = f"Scatter Plot of {', '.join(y_column_names)} vs {x_label}"

    for col in y_column_names:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[col],
                mode=mode,
                name=col,
                marker=dict(size=6),
                line=dict(width=2),
                hovertemplate=f"<b>{col}</b><br>{x_label}: %{{x}}<br>{col}: %{{y}}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title=x_label,
        yaxis_title="Values",
        template="plotly_white",
        legend=dict(title="Y Columns", orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def scatter_plot_series(
        series: pd.Series,
        title: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()

    if title is None:
        title = f"Scatter Plot of {series.name} vs Index"

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="markers+lines",
            name=series.name,
            marker=dict(size=6),
            line=dict(width=2),
            hovertemplate=f"<b>{series.name}</b><br>Index: %{{x}}<br>Value: %{{y}}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Index",
        yaxis_title="Values",
        template="plotly_white",
        legend=dict(title="Series", orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    color: Optional[str] = None,
    nbins: int = 40,
    plot_probability: bool = False,
    title: Optional[str] = None,
):
    if title is None:
        title = f"Histogram of {column}"

    fig = go.Figure()
    histnorm = None if not plot_probability else "probability"

    if color and color in df.columns:
        for val in df[color].dropna().unique():
            subset = df[df[color] == val]
            fig.add_trace(
                go.Histogram(
                    x=subset[column],
                    nbinsx=nbins,
                    name=f"{color} = {val}",
                    opacity=0.6,
                    histnorm=histnorm,
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=df[column],
                nbinsx=nbins,
                name=column,
                opacity=0.6,
                histnorm=histnorm,
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        title=title,
        xaxis_title=column,
        yaxis_title="Count" if not plot_probability else "Probability",
    )

    return fig


def plot_feature_correlation_heatmap(df: pd.DataFrame, title="Feature Correlation Heatmap"):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    plt.title(title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


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


def plot_uplift_at_k_trend(
        y_retention_test: pd.Series,
        uplift_retention_predictions_test: pd.Series,
        treatment_test: pd.Series,
) -> go.Figure:
    ks = np.arange(0.05, 1.0, 0.05)
    uplift_values = []

    for k in tqdm(ks, desc='Computing uplift@k'):
        val = uplift_at_k(
            y_true=y_retention_test,
            uplift=uplift_retention_predictions_test,
            treatment=treatment_test,
            strategy='by_group',
            k=k if k > 0 else 0.001
        )
        uplift_values.append(val)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ks * 100,
        y=uplift_values,
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=8),
        name='Uplift@k'
    ))

    fig.update_layout(
        title='Uplift@k Trend',
        xaxis_title='Top-k [%]',
        yaxis_title='Uplift@k',
        template='plotly_white',
        hovermode='x unified',
        title_x=0.5,
    )

    return fig
