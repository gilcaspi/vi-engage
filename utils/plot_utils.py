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
        x_column_name: str,
        y_column_names: List[str],
        title: str,
        mode: str = "markers+lines"
) -> go.Figure:
    fig = go.Figure()

    for col in y_column_names:
        fig.add_trace(
            go.Scatter(
                x=df[x_column_name],
                y=df[col],
                mode=mode,
                name=col,
                marker=dict(size=6),
                line=dict(width=2),
                hovertemplate=f"<b>{col}</b><br>{x_column_name}: %{{x}}<br>{col}: %{{y}}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title=x_column_name,
        yaxis_title="Values",
        template="plotly_white",
        legend=dict(title="Y Columns", orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig