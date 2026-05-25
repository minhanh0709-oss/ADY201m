"""CLV & VIP Targeting dashboard — Plotly Dash entry point.

Usage:
    python dashboard/app.py

Open http://127.0.0.1:8050 in a browser.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (python dashboard/app.py) by exposing the
# project root on sys.path so the package import "dashboard.pages" works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from dashboard.pages import (page1_overview, page2_rfm_matrix,
                             page3_clv_forecast, page4_campaign_targeting)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="CLV & VIP Targeting Dashboard",
    suppress_callback_exceptions=True,
)

nav_items = [
    ("Overview",          "/",          "page-1"),
    ("RFM Matrix",        "/rfm",       "page-2"),
    ("CLV Forecast",      "/forecast",  "page-3"),
    ("Campaign Targeting","/campaign",  "page-4"),
]

navbar = dbc.NavbarSimple(
    children=[dbc.NavItem(dbc.NavLink(name, href=href, id=f"nav-{nid}",
                                       active="exact"))
              for (name, href, nid) in nav_items],
    brand="CLV & VIP Targeting",
    brand_href="/", color="primary", dark=True, sticky="top",
    fluid=True,
)

footer = html.Footer(
    dbc.Container(
        html.Small("ADY201m Project · Online Retail II · Window 3 "
                   "(18-month obs / 6-month pred) · Hurdle + CQR predictions",
                   className="text-muted"),
        className="py-3 text-center",
    ),
    className="mt-4 border-top",
)

app.layout = dbc.Container([
    dcc.Location(id="url"),
    navbar,
    dbc.Container(id="page-content", className="mt-4", fluid=True),
    footer,
], fluid=True, className="px-0")


@app.callback(Output("page-content", "children"),
              Input("url", "pathname"))
def render(pathname: str):
    if pathname in ("/", None, ""):
        return page1_overview.layout()
    if pathname == "/rfm":
        return page2_rfm_matrix.layout()
    if pathname == "/forecast":
        return page3_clv_forecast.layout()
    if pathname == "/campaign":
        return page4_campaign_targeting.layout()
    return dbc.Alert(
        f"Page not found: {pathname}",
        color="danger", className="mt-3")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
