from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go


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


def plot_histograms(
        df: pd.DataFrame,
        columns: list[str],
        color: Optional[str] = None,
        nbins: int = 40,
        plot_probability: bool = False,
        title: Optional[str] = None,
):
    if title is None:
        title = "Histogram of " + ", ".join(columns)

    fig = go.Figure()
    histnorm = None if not plot_probability else "probability"
    for col in columns:
        if color and color in df.columns:
            for val in df[color].dropna().unique():
                subset = df[df[color] == val]
                fig.add_trace(
                    go.Histogram(x=subset[col], nbinsx=nbins, name=f"{col} - {val}", opacity=0.6, histnorm=histnorm)
                )
        else:
            fig.add_trace(go.Histogram(x=df[col], nbinsx=nbins, name=col, opacity=0.6, histnorm=histnorm))
    fig.update_layout(barmode="overlay", template="plotly_white", title=title)
    return fig