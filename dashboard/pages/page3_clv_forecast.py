"""Page 3 — CLV Forecast: top-100 predicted VIPs with CQR intervals,
calibration decile chart, predicted-vs-actual scatter."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from .. import data_loader as dl


def _top100_table():
    pred = dl.predictions().head(100).copy()
    pred["rank"] = range(1, len(pred) + 1)
    pred["CQR_Width"] = pred["CQR_Upper"] - pred["CQR_Lower"]
    return dash_table.DataTable(
        id="top100-table",
        columns=[
            {"name": "#", "id": "rank"},
            {"name": "Customer", "id": "CustomerID"},
            {"name": "Country",  "id": "Country"},
            {"name": "Pred CLV ($)", "id": "PredictedCLV",
             "type": "numeric",
             "format": dash_table.Format.Format(precision=2,
                          scheme=dash_table.Format.Scheme.fixed)},
            {"name": "CQR Lo",   "id": "CQR_Lower",
             "type": "numeric",
             "format": dash_table.Format.Format(precision=0,
                          scheme=dash_table.Format.Scheme.fixed)},
            {"name": "CQR Hi",   "id": "CQR_Upper",
             "type": "numeric",
             "format": dash_table.Format.Format(precision=0,
                          scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Actual",   "id": "ActualCLV",
             "type": "numeric",
             "format": dash_table.Format.Format(precision=2,
                          scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Segment",  "id": "RFM_Segment"},
        ],
        data=pred.to_dict("records"),
        page_size=20,
        sort_action="native",
        style_table={"height": "520px", "overflowY": "auto"},
        style_cell={"fontSize": "0.82rem",
                    "fontFamily": "Segoe UI, sans-serif"},
        style_header={"backgroundColor": "#e9ecef", "fontWeight": "600"},
        style_data_conditional=[
            {"if": {"filter_query": "{ActualCLV} > {PredictedCLV}",
                    "column_id": "ActualCLV"},
             "backgroundColor": "#d4edda", "color": "#155724"},
        ],
    )


def _error_bar_chart():
    pred = dl.predictions().head(50).copy()
    pred["rank"] = range(1, len(pred) + 1)
    pred["err_lo"] = pred["PredictedCLV"] - pred["CQR_Lower"]
    pred["err_hi"] = pred["CQR_Upper"]   - pred["PredictedCLV"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["rank"], y=pred["PredictedCLV"],
        mode="markers",
        marker=dict(size=8, color="#1f77b4", symbol="circle"),
        name="Predicted CLV",
        error_y=dict(type="data",
                     array=pred["err_hi"], arrayminus=pred["err_lo"],
                     thickness=1.2, width=3, color="#aac"),
    ))
    fig.add_trace(go.Scatter(
        x=pred["rank"], y=pred["ActualCLV"],
        mode="markers",
        marker=dict(size=8, color="#d62728", symbol="x"),
        name="Actual CLV",
    ))
    fig.update_layout(
        title="Top-50 predicted VIPs with 95% CQR intervals",
        xaxis_title="Rank by predicted CLV",
        yaxis_title="CLV in prediction window ($)",
        height=460, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee", type="log")
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def _decile_chart():
    d = dl.decile()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["decile"], y=d["actual_mean"],
                         name="Actual mean", marker_color="#1f77b4"))
    fig.add_trace(go.Scatter(x=d["decile"], y=d["pred_mean"],
                             name="Predicted mean", mode="lines+markers",
                             marker_color="#d62728"))
    fig.update_layout(
        title="Calibration by predicted-CLV decile",
        xaxis_title="Decile (1=lowest predicted)",
        yaxis_title="Mean CLV ($)",
        height=460, plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def _scatter_pred_vs_actual():
    pred = dl.predictions().copy()
    pred = pred[pred["ActualCLV"] > 0]
    fig = px.scatter(
        pred, x="PredictedCLV", y="ActualCLV", color="RFM_Segment",
        hover_data=["CustomerID", "Country", "Monetary"],
        labels={"PredictedCLV": "Predicted CLV ($)",
                "ActualCLV":    "Actual CLV ($)"},
        title="Predicted vs Actual (non-zero customers, Window 3)",
        log_x=True, log_y=True, opacity=0.6,
    )
    mx = max(pred["PredictedCLV"].max(), pred["ActualCLV"].max())
    fig.add_shape(type="line", x0=1, y0=1, x1=mx, y1=mx,
                  line=dict(color="gray", dash="dash"))
    fig.update_layout(height=460, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=60, b=10))
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def layout() -> html.Div:
    return html.Div([
        html.H3("CLV Forecast & Uncertainty", className="page-title"),
        html.P("Hurdle model predictions on Window 3 (18-month observation / "
               "6-month prediction). CQR intervals provide 95% coverage on "
               "held-out customers.", className="page-subtitle"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top-100 predicted VIPs"),
                _top100_table(),
            ])), md=12, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(_error_bar_chart())), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(_decile_chart())), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(_scatter_pred_vs_actual())),
                    md=12),
        ]),
    ])
