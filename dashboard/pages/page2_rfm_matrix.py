"""Page 2 — RFM Matrix: 5x5 heatmap with click-to-drill + CLV violin."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

from .. import data_loader as dl


def _heatmap(grid):
    pivot_n      = grid.pivot(index="R_q", columns="F_q", values="n").fillna(0)
    pivot_mean_m = grid.pivot(index="R_q", columns="F_q", values="mean_M").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot_mean_m.values,
        x=[f"F{c}" for c in pivot_mean_m.columns],
        y=[f"R{r} (best={5})" if r == 5 else f"R{r}"
           for r in pivot_mean_m.index],
        colorscale="Viridis", colorbar=dict(title="Mean Monetary ($)"),
        text=pivot_n.values.astype(int),
        texttemplate="%{text}",  hoverinfo="x+y+z+text",
    ))
    fig.update_layout(
        title="RFM 5x5 grid — color = mean Monetary, cell text = # customers",
        xaxis_title="Frequency quintile (1=low, 5=high)",
        yaxis_title="Recency quintile (1=most recent, 5=oldest)",
        height=480, margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="white",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _segment_violin(df):
    order = ["Champion", "Loyal", "At-Risk", "Lost", "New"]
    fig = px.violin(
        df, x="RFM_Segment", y="ActualCLV",
        category_orders={"RFM_Segment": order},
        color="RFM_Segment", points=False, box=True,
        title="Future CLV distribution by RFM segment (Window 3 test fold)",
        labels={"ActualCLV": "Actual CLV in prediction window ($)"},
    )
    fig.update_layout(showlegend=False, height=420,
                      margin=dict(l=10, r=10, t=60, b=10),
                      plot_bgcolor="white")
    return fig


def layout() -> html.Div:
    grid, full = dl.rfm_matrix()

    # Top-20 by Monetary for the empty-state drill table
    default_rows = (dl.rfm_segments()
                    .sort_values("Monetary", ascending=False)
                    .head(20)
                    [["CustomerID", "RFM_Segment", "Recency", "Frequency", "Monetary",
                      "AvgOrderValue", "Country"]]
                    .to_dict("records"))

    return html.Div([
        html.H3("RFM Customer Matrix", className="page-title"),
        html.P("Click any cell to see the top-20 customers in that "
               "Recency × Frequency quintile bucket.",
               className="page-subtitle"),

        dcc.Store(id="rfm-store",
                  data=full[["CustomerID", "Recency", "Frequency",
                             "Monetary", "AvgOrderValue", "Country",
                             "R_q", "F_q", "RFM_Segment"]].to_dict("records")),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id="rfm-heatmap", figure=_heatmap(grid),
                          config={"displaylogo": False})
            )), md=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(id="drill-title",
                        children="Top-20 customers (click a cell to filter)"),
                dash_table.DataTable(
                    id="drill-table",
                    columns=[
                        {"name": "Customer", "id": "CustomerID"},
                        {"name": "Segment", "id": "RFM_Segment"},
                        {"name": "R", "id": "Recency"},
                        {"name": "F", "id": "Frequency"},
                        {"name": "Monetary",  "id": "Monetary",
                         "type": "numeric",
                         "format": dash_table.Format.Format(precision=2,
                                       scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "AOV", "id": "AvgOrderValue",
                         "type": "numeric",
                         "format": dash_table.Format.Format(precision=2,
                                       scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Country", "id": "Country"},
                    ],
                    data=default_rows,
                    page_size=20,
                    style_table={"height": "400px", "overflowY": "auto"},
                    style_cell={"fontSize": "0.85rem",
                                "fontFamily": "Segoe UI, sans-serif"},
                    style_header={"backgroundColor": "#e9ecef",
                                  "fontWeight": "600"},
                ),
            ])), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=_segment_violin(dl.predictions()),
                          config={"displaylogo": False})
            )), md=12),
        ]),
    ])


# -------- callbacks --------
@callback(
    Output("drill-table", "data"),
    Output("drill-title", "children"),
    Input("rfm-heatmap", "clickData"),
    State("rfm-store", "data"),
    prevent_initial_call=True,
)
def _drill(click_data, store):
    if not click_data:
        return [], "Top-20 customers (click a cell to filter)"
    import pandas as pd
    pt = click_data["points"][0]
    fq = int(pt["x"].lstrip("F"))
    ry = pt["y"]
    rq = int(ry.split()[0].lstrip("R"))
    df = pd.DataFrame(store)
    sel = df[(df["R_q"] == rq) & (df["F_q"] == fq)]
    sel = sel.sort_values("Monetary", ascending=False).head(20)
    rows = sel[["CustomerID", "RFM_Segment", "Recency", "Frequency", "Monetary",
                "AvgOrderValue", "Country"]].to_dict("records")
    title = f"Top-20 customers in cell (R={rq}, F={fq}) — {len(sel)} of {len(df[(df['R_q']==rq)&(df['F_q']==fq)])} match"
    return rows, title
