"""Page 4 — Campaign Targeting: live K%-slider profit simulation,
semantic-cluster filter, and CSV export of the targeted customer list."""
from __future__ import annotations

import io

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

from .. import data_loader as dl

K_DEFAULT     = 10
MARGIN_DEFAULT = 20.0  # %
COST_DEFAULT   = 3.0   # $


def _kpi(label, id_, sub=""):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(label, className="kpi-label"),
            html.Div(id=id_, className="kpi-value"),
            html.Div(sub, className="text-muted small"),
        ]), className="kpi-card"),
        md=3, className="mb-3",
    )


def _semantic_options():
    stats = dl.sem_cluster_stats()
    products = dl.sem_cluster_products()
    opts = [{"label": "All clusters", "value": -1}]
    for c in sorted(stats["sem_cluster"].unique()):
        top_prod = (products[products["cluster"] == c]
                    .head(1)["product"].values)
        top = top_prod[0][:25] if len(top_prod) else f"cluster {c}"
        opts.append({"label": f"Cluster {c} — {top}", "value": int(c)})
    return opts


def layout() -> html.Div:
    return html.Div([
        html.H3("Campaign Targeting & Profit Simulation",
                className="page-title"),
        html.P("Interactive contact-rate slider with live profit, revenue "
               "capture, and CSV export of the targeted customer list. "
               "Numbers are computed on the Window 3 test fold.",
               className="page-subtitle"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Contact rate K (%)"),
                dcc.Slider(id="k-slider", min=1, max=50, step=1,
                           value=K_DEFAULT,
                           marks={i: str(i) for i in
                                  [1, 5, 10, 15, 20, 25, 30, 40, 50]}),
            ], md=6),
            dbc.Col([
                dbc.Label("Gross margin (%)"),
                dcc.Slider(id="margin-slider", min=5, max=50, step=1,
                           value=MARGIN_DEFAULT,
                           marks={i: str(i) for i in [5, 10, 20, 30, 40, 50]}),
            ], md=3),
            dbc.Col([
                dbc.Label("Contact cost ($/customer)"),
                dcc.Slider(id="cost-slider", min=0, max=10, step=0.5,
                           value=COST_DEFAULT,
                           marks={i: f"${i}" for i in [0, 2, 5, 8, 10]}),
            ], md=3),
        ], className="mb-4"),

        dbc.Row([
            _kpi("Customers contacted",  "kpi-contacts"),
            _kpi("Revenue captured",     "kpi-revenue"),
            _kpi("Net profit",           "kpi-profit"),
            _kpi("Lift vs Random",       "kpi-lift"),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Label("Filter by semantic taste cluster"),
                dcc.Dropdown(id="sem-filter",
                             options=_semantic_options(),
                             value=-1, clearable=False),
            ], md=4),
            dbc.Col([
                html.Br(),
                dbc.Button("Download targeted list (CSV)",
                           id="btn-export",
                           color="primary"),
                dcc.Download(id="export-csv"),
            ], md=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id="rc-curve", config={"displaylogo": False})
            )), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id="cluster-bar", config={"displaylogo": False})
            )), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(id="target-title"),
                dash_table.DataTable(
                    id="target-table",
                    columns=[
                        {"name": "Customer", "id": "CustomerID"},
                        {"name": "Country",  "id": "Country"},
                        {"name": "Pred CLV",  "id": "PredictedCLV",
                         "type": "numeric",
                         "format": dash_table.Format.Format(precision=2,
                                      scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Actual",    "id": "ActualCLV",
                         "type": "numeric",
                         "format": dash_table.Format.Format(precision=2,
                                      scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Segment",   "id": "RFM_Segment"},
                        {"name": "Sem cluster", "id": "SemCluster"},
                    ],
                    page_size=15,
                    style_cell={"fontSize": "0.82rem"},
                    style_header={"backgroundColor": "#e9ecef",
                                  "fontWeight": "600"},
                ),
            ])), md=12),
        ]),
    ])


# -------- callbacks --------
def _filter_predictions(sem_cluster: int) -> pd.DataFrame:
    df = dl.predictions().copy()
    if sem_cluster != -1:
        df = df[df["SemCluster"] == sem_cluster]
    return df.sort_values("PredictedCLV", ascending=False).reset_index(drop=True)


@callback(
    Output("kpi-contacts", "children"),
    Output("kpi-revenue",  "children"),
    Output("kpi-profit",   "children"),
    Output("kpi-lift",     "children"),
    Output("rc-curve",     "figure"),
    Output("cluster-bar",  "figure"),
    Output("target-table", "data"),
    Output("target-title", "children"),
    Input("k-slider",      "value"),
    Input("margin-slider", "value"),
    Input("cost-slider",   "value"),
    Input("sem-filter",    "value"),
)
def _update(k_pct, margin_pct, cost, sem):
    df = _filter_predictions(sem)
    n = len(df)
    n_target = max(1, int(round(k_pct / 100 * n)))
    top = df.head(n_target)

    total_rev = float(df["ActualCLV"].sum())
    capt_rev  = float(top["ActualCLV"].sum())
    rc_pct    = 100 * capt_rev / total_rev if total_rev > 0 else 0
    profit    = (margin_pct / 100) * capt_rev - cost * n_target
    random_rc = k_pct
    lift_x    = rc_pct / random_rc if random_rc > 0 else 0

    kpi_contacts = f"{n_target:,} / {n:,}"
    kpi_revenue  = f"${capt_rev/1e3:,.1f} k  ({rc_pct:.1f}%)"
    kpi_profit   = f"${profit:,.0f}"
    kpi_lift     = f"{lift_x:.2f}×"

    # Revenue capture curve
    df_sorted = df.sort_values("PredictedCLV", ascending=False).reset_index(drop=True)
    df_sorted["cum_actual"] = df_sorted["ActualCLV"].cumsum() / total_rev * 100
    df_sorted["rank_pct"]   = (df_sorted.index + 1) / n * 100
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(x=df_sorted["rank_pct"],
                                y=df_sorted["cum_actual"],
                                name="Hurdle", line=dict(color="#1f77b4")))
    fig_rc.add_trace(go.Scatter(x=[0, 100], y=[0, 100],
                                name="Random",
                                line=dict(color="gray", dash="dash")))
    fig_rc.add_vline(x=k_pct, line_color="#d62728", line_dash="dot",
                     annotation_text=f"K={k_pct}%")
    fig_rc.update_layout(
        title=f"Revenue Capture curve (current K={k_pct}%)",
        xaxis_title="% Customers contacted (sorted by predicted CLV)",
        yaxis_title="% Revenue captured",
        height=440, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
    fig_rc.update_xaxes(range=[0, 100], gridcolor="#eee")
    fig_rc.update_yaxes(range=[0, 100], gridcolor="#eee")

    # Cluster budget allocation
    by_cluster = (top.groupby("SemCluster")
                     .agg(n=("CustomerID", "count"),
                          rev=("ActualCLV", "sum"))
                     .reset_index())
    by_cluster["SemCluster"] = by_cluster["SemCluster"].astype(str)
    fig_cl = px.bar(by_cluster, x="SemCluster", y="n",
                    color="rev", color_continuous_scale="Blues",
                    title=f"Targeted customers by semantic cluster (top {k_pct}%)",
                    labels={"n": "# targeted",
                            "rev": "Actual revenue ($)",
                            "SemCluster": "Cluster id"})
    fig_cl.update_layout(height=440, plot_bgcolor="white",
                         margin=dict(l=10, r=10, t=60, b=10))
    fig_cl.update_xaxes(gridcolor="#eee")
    fig_cl.update_yaxes(gridcolor="#eee")

    # Table
    table = top[["CustomerID", "Country", "PredictedCLV",
                 "ActualCLV", "RFM_Segment", "SemCluster"]].to_dict("records")
    title = (f"Targeted customer list — top {k_pct}% by predicted CLV "
             f"({n_target:,} of {n:,} customers)")

    return (kpi_contacts, kpi_revenue, kpi_profit, kpi_lift,
            fig_rc, fig_cl, table, title)


@callback(
    Output("export-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("k-slider",   "value"),
    State("sem-filter", "value"),
    prevent_initial_call=True,
)
def _export(_, k_pct, sem):
    df = _filter_predictions(sem)
    n_target = max(1, int(round(k_pct / 100 * len(df))))
    top = df.head(n_target)
    buf = io.StringIO()
    top.to_csv(buf, index=False)
    return dict(content=buf.getvalue(),
                filename=f"targeted_top{k_pct}pct.csv")
