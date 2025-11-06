from typing import List, Optional

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
