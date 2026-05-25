"""Page 1 — Overview: top KPIs, MAC + revenue trend, country treemap."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from .. import data_loader as dl


def _kpi(label: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sub, className="text-muted small"),
            ]),
            className="kpi-card",
        ),
        md=3, className="mb-3",
    )


def _kpi_row() -> dbc.Row:
    k = dl.kpi_summary()
    return dbc.Row([
        _kpi("Total Revenue", f"${k['total_revenue']/1e6:,.2f} M",
             f"{k['total_invoices']:,} invoices"),
        _kpi("Total Customers", f"{k['total_customers']:,}",
             "with valid CustomerID"),
        _kpi("Avg Order Value", f"${k['avg_order_value']:,.2f}",
             "per customer mean"),
        _kpi("90-Day Active Rate", f"{k['retention_90d']:.1f}%",
             "Recency ≤ 90 days"),
    ])


def _monthly_chart() -> dcc.Graph:
    mac = dl.mac()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=mac["year_month"], y=mac["revenue"],
               name="Revenue", marker_color="#1f77b4", opacity=0.55),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=mac["year_month"], y=mac["active_customers"],
                   name="Active customers", mode="lines+markers",
                   line=dict(color="#d62728", width=2.5)),
        secondary_y=False,
    )
    fig.update_layout(
        title="Monthly Active Customers & Revenue",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.18),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Month",
                     gridcolor="#eee", showline=True, linecolor="#ccc")
    fig.update_yaxes(title_text="Active customers", secondary_y=False,
                     gridcolor="#eee")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=True,
                     showgrid=False)
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def _country_treemap() -> dcc.Graph:
    c = dl.country().head(15)
    fig = px.treemap(
        c, path=[px.Constant("All"), "region", "Country"],
        values="total_revenue", color="revenue_per_customer",
        color_continuous_scale="Blues",
        hover_data={"n_customers": True, "pct_of_revenue": ":.2f"},
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title="Revenue by Country (UK vs Non-UK)",
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def _retention_strip() -> dcc.Graph:
    """Compact retention heat-map (first 12 months of each cohort)."""
    coh = dl.cohort().copy()
    coh = coh[coh["month_offset"] <= 12]
    sizes = (coh.query("month_offset == 0")
                .set_index("cohort_month")["active_customers"])
    coh["retention_pct"] = (100 * coh["active_customers"]
                            / coh["cohort_month"].map(sizes))
    pivot = coh.pivot(index="cohort_month", columns="month_offset",
                      values="retention_pct")
    fig = px.imshow(pivot.values,
                    x=pivot.columns,
                    y=[d.strftime("%Y-%m") for d in pivot.index],
                    color_continuous_scale="YlOrRd", aspect="auto",
                    labels=dict(x="Months after first purchase",
                                y="Cohort", color="Retention %"))
    fig.update_layout(title="Cohort Retention (first 12 months)",
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def layout() -> html.Div:
    return html.Div([
        html.H3("Business Overview", className="page-title"),
        html.P("Top-level KPIs and temporal patterns from Online Retail II "
               "(2009-12 to 2011-12). All numbers are post-cleaning.",
               className="page-subtitle"),

        _kpi_row(),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(_monthly_chart())), md=12,
                    className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(_country_treemap())), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(_retention_strip())), md=6),
        ], className="mb-3"),
    ])
