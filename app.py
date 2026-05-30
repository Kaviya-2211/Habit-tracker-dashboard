

# ── CELL 1: Install dependencies ─────────────────────────────
# Run this first, then restart runtime if asked




# ── CELL 2: Imports ──────────────────────────────────────────

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, date
import calendar
import json



# ── CELL 3: Default Habits & Config ──────────────────────────

DEFAULT_HABITS = [
    "Wakeup at 5 AM",
    "Drink chia seeds water",
    "Exercise for 1 hour",
    "Walk 30 min after breakfast",
    "Study for 7 hours",
    "Extra skill for 1 hour",
    "No sugar & no junk food",
    "Eating diet food",
    "Walk 10,000 steps daily",
    "Drink 3 litres water",
    "Proper skin care",
    "Revision for 1 hour",
    "Sleep at 10 PM",
    "Relaxation",
    "Keep bed clean",
    "Mid nap in afternoon (30 min)",
    "No mobile after 10 PM",
    "12 hours fasting",
]

COLORS = {
    "bg":        "#0a0a14",
    "card":      "#111122",
    "border":    "#1e1e3a",
    "green":     "#00ff87",
    "blue":      "#60efff",
    "yellow":    "#ffe45e",
    "purple":    "#a29bfe",
    "red":       "#ff6b6b",
    "orange":    "#ff9f43",
    "text":      "#e0e0f0",
    "muted":     "#555566",
}


# ── CELL 4: ML Model ─────────────────────────────────────────

def run_ml_model(habits, data, year, month):
    """
    ML model that computes per-habit statistics:
    - Consistency rate
    - Current & max streak
    - Day-of-week pattern (best/worst day)
    - Momentum (last 7 vs prev 7 days)
    - End-of-month prediction
    - Composite ML score (0-100)
    """
    days_in_month = calendar.monthrange(year, month)[1]
    today = date.today()
    is_current = (today.year == year and today.month == month)
    today_day = today.day if is_current else days_in_month

    results = []
    day_names = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]

    for habit in habits:
        row = data.get(habit, {})
        completed = 0
        streak = 0
        max_streak = 0
        cur = 0
        dow_counts = [0]*7
        dow_total  = [0]*7

        for d in range(1, days_in_month + 1):
            dt = date(year, month, d)
            dow = dt.weekday()          # Mon=0 … Sun=6
            dow_total[dow] += 1
            done = row.get(str(d), False)
            if done:
                completed += 1
                cur += 1
                dow_counts[dow] += 1
                if d <= today_day:
                    max_streak = max(max_streak, cur)
            else:
                if d < today_day:
                    cur = 0

        # current streak (backwards from today)
        for d in range(today_day, 0, -1):
            if row.get(str(d), False):
                streak += 1
            else:
                break

        days_elapsed = min(today_day, days_in_month)
        consistency  = completed / days_elapsed if days_elapsed > 0 else 0

        # day-of-week affinity
        dow_scores = [dow_counts[i]/dow_total[i] if dow_total[i] > 0 else 0 for i in range(7)]
        best_dow  = int(np.argmax(dow_scores))
        worst_dow = int(np.argmin(dow_scores))

        # momentum: last 7 vs previous 7
        last7 = sum(1 for d in range(max(1, today_day-6), today_day+1) if row.get(str(d), False))
        prev7 = sum(1 for d in range(max(1, today_day-13), max(1, today_day-6)+1) if row.get(str(d), False))
        momentum = last7 - prev7

        # end-of-month prediction
        remaining = days_in_month - today_day
        predicted_total = completed + round(consistency * remaining)
        predicted_rate  = predicted_total / days_in_month if days_in_month > 0 else 0

        # composite ML score
        ml_score = min(100, round(
            consistency * 50 +
            (streak / max(1, days_elapsed)) * 20 +
            max(0, momentum / 7) * 20 +
            (max_streak / max(1, days_elapsed)) * 10
        ))

        results.append({
            "habit":          habit,
            "completed":      completed,
            "streak":         streak,
            "max_streak":     max_streak,
            "consistency":    round(consistency * 100, 1),
            "momentum":       momentum,
            "predicted_rate": round(predicted_rate * 100, 1),
            "ml_score":       ml_score,
            "best_day":       day_names[best_dow],
            "worst_day":      day_names[worst_dow],
            "days_elapsed":   days_elapsed,
            "days_in_month":  days_in_month,
        })

    return pd.DataFrame(results)


def get_insight(row):
    if row["consistency"] >= 90:
        return "🔥 On fire! Keep this streak alive."
    if row["momentum"] > 2:
        return "📈 Momentum building — you're improving!"
    if row["streak"] >= 5:
        return f"⚡ {int(row['streak'])}-day streak! Don't break it."
    if row["momentum"] < -2:
        return "⚠️ Slipping — refocus today."
    if row["consistency"] < 40:
        return "💡 Low consistency. Try habit stacking."
    return "🎯 Steady. Push for 80%+ this month."


# ── CELL 5: Chart Builders ────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="monospace"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def score_gauge(value, title="ML SCORE"):
    color = COLORS["green"] if value >= 75 else COLORS["yellow"] if value >= 50 else COLORS["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=12, color=COLORS["muted"])),
        number=dict(font=dict(size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor=COLORS["muted"], tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=COLORS["border"],
            steps=[
                dict(range=[0,40],  color="#1a0a0a"),
                dict(range=[40,75], color="#0a0a1a"),
                dict(range=[75,100],color="#0a1a0a"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=value)
        )
    ))
    fig.update_layout(**PLOT_LAYOUT, height=200)
    return fig


def heatmap_chart(df_stats, data, year, month):
    days_in_month = calendar.monthrange(year, month)[1]
    habits = df_stats["habit"].tolist()
    matrix = []
    for habit in habits:
        row_data = data.get(habit, {})
        matrix.append([1 if row_data.get(str(d), False) else 0 for d in range(1, days_in_month+1)])

    today = date.today()
    is_current = (today.year == year and today.month == month)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=list(range(1, days_in_month+1)),
        y=[h[:28]+"…" if len(h)>28 else h for h in habits],
        colorscale=[[0,"#111122"],[1, COLORS["green"]]],
        showscale=False,
        hovertemplate="Day %{x}<br>%{y}<br>Done: %{z}<extra></extra>",
        xgap=2, ygap=2,
    ))

    # today line
    if is_current and 1 <= today.day <= days_in_month:
        fig.add_vline(x=today.day, line_color=COLORS["blue"], line_width=1.5, opacity=0.6)

    fig.update_layout(
        **PLOT_LAYOUT,
        height=max(300, len(habits)*32 + 60),
        xaxis=dict(title="Day of Month", tickcolor=COLORS["muted"], gridcolor=COLORS["border"]),
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        title=dict(text="Monthly Habit Grid", font=dict(size=13, color=COLORS["muted"])),
    )
    return fig


def bar_consistency(df_stats):
    df_sorted = df_stats.sort_values("consistency", ascending=True)
    colors = [
        COLORS["green"]  if c >= 80 else
        COLORS["yellow"] if c >= 50 else
        COLORS["red"]
        for c in df_sorted["consistency"]
    ]
    fig = go.Figure(go.Bar(
        x=df_sorted["consistency"],
        y=[h[:25]+"…" if len(h)>25 else h for h in df_sorted["habit"]],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c}%" for c in df_sorted["consistency"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="%{y}<br>Consistency: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=max(300, len(df_stats)*28 + 80),
        xaxis=dict(range=[0,115], title="Consistency %", gridcolor=COLORS["border"]),
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        title=dict(text="Consistency by Habit", font=dict(size=13, color=COLORS["muted"])),
    )
    return fig


def radar_chart(df_stats):
    top = df_stats.nlargest(8, "ml_score")
    categories = [h[:18]+"…" if len(h)>18 else h for h in top["habit"]]
    values = top["ml_score"].tolist()
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor=f"rgba(0,255,135,0.15)",
        line=dict(color=COLORS["green"], width=2),
        marker=dict(color=COLORS["green"], size=6),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(range=[0,100], tickcolor=COLORS["muted"],
                            gridcolor=COLORS["border"], tickfont=dict(size=8)),
            angularaxis=dict(tickcolor=COLORS["muted"], gridcolor=COLORS["border"],
                             tickfont=dict(size=9)),
        ),
        title=dict(text="Top Habits Radar (ML Score)", font=dict(size=13, color=COLORS["muted"])),
    )
    return fig


def momentum_chart(df_stats):
    df = df_stats.sort_values("momentum")
    colors = [COLORS["green"] if m >= 0 else COLORS["red"] for m in df["momentum"]]
    fig = go.Figure(go.Bar(
        x=[h[:22]+"…" if len(h)>22 else h for h in df["habit"]],
        y=df["momentum"],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{x}<br>Momentum: %{y}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=280,
        xaxis=dict(tickangle=-35, tickfont=dict(size=9), gridcolor=COLORS["border"]),
        yaxis=dict(title="Momentum (last 7 - prev 7 days)", gridcolor=COLORS["border"]),
        title=dict(text="Habit Momentum", font=dict(size=13, color=COLORS["muted"])),
    )
    return fig


def prediction_chart(df_stats):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current %",
        x=[h[:20]+"…" if len(h)>20 else h for h in df_stats["habit"]],
        y=df_stats["consistency"],
        marker=dict(color=COLORS["blue"], opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        name="Predicted EOMonth %",
        x=[h[:20]+"…" if len(h)>20 else h for h in df_stats["habit"]],
        y=df_stats["predicted_rate"],
        mode="markers+lines",
        marker=dict(color=COLORS["yellow"], size=8, symbol="diamond"),
        line=dict(color=COLORS["yellow"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=300,
        xaxis=dict(tickangle=-35, tickfont=dict(size=9), gridcolor=COLORS["border"]),
        yaxis=dict(range=[0,110], title="%", gridcolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        title=dict(text="Current vs Predicted End-of-Month", font=dict(size=13, color=COLORS["muted"])),
        barmode="group",
    )
    return fig


# ── CELL 6: Dash App Layout ───────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)

# Shared style helpers
CARD = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "16px",
}
LABEL = {"fontSize": "10px", "color": COLORS["muted"], "letterSpacing": "2px", "textTransform": "uppercase"}
BIG   = {"fontSize": "28px", "fontWeight": "700", "fontFamily": "monospace"}

def stat_card(title, value, color, subtitle=""):
    return html.Div([
        html.Div(title, style=LABEL),
        html.Div(value, style={**BIG, "color": color}),
        html.Div(subtitle, style={"fontSize": "11px", "color": COLORS["muted"]}),
    ], style={**CARD, "textAlign": "center", "padding": "20px 12px"})


app.layout = html.Div(style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
                              "fontFamily": "monospace", "color": COLORS["text"]}, children=[

    dcc.Store(id="store-data",   data={}),
    dcc.Store(id="store-habits", data=DEFAULT_HABITS),

    # ── HEADER ──────────────────────────────────────────────
    html.Div(style={"background": "#0d0d1f", "borderBottom": f"1px solid {COLORS['border']}",
                    "padding": "20px 32px"}, children=[
        dbc.Row([
            dbc.Col([
                html.Div("ML HABIT TRACKER", style={**LABEL, "marginBottom": "4px"}),
                html.H2("Monthly Tracker", style={"color": COLORS["text"], "margin": 0, "letterSpacing": "-1px"}),
            ], width=4),
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id="dd-month",
                        options=[{"label": m, "value": i+1} for i, m in enumerate(
                            ["January","February","March","April","May","June",
                             "July","August","September","October","November","December"])],
                        value=date.today().month,
                        clearable=False,
                        style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
                    ), width=6),
                    dbc.Col(dcc.Dropdown(
                        id="dd-year",
                        options=[{"label": str(y), "value": y} for y in range(2024, 2027)],
                        value=date.today().year,
                        clearable=False,
                        style={"backgroundColor": COLORS["card"]},
                    ), width=6),
                ])
            ], width=4),
            dbc.Col([
                dbc.Input(id="input-habit", placeholder="Add new habit…",
                          style={"background": COLORS["card"], "color": COLORS["text"],
                                 "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=3),
            dbc.Col(
                dbc.Button("+ Add", id="btn-add-habit", color="success", size="sm",
                           style={"background": COLORS["green"], "border": "none",
                                  "color": "#0a0a14", "fontWeight": "700"}),
                width=1
            ),
        ], align="center"),
    ]),

    # ── TABS ────────────────────────────────────────────────
    html.Div(style={"padding": "0 32px"}, children=[
        dcc.Tabs(id="tabs", value="grid", style={"borderBottom": f"1px solid {COLORS['border']}"},
                 colors={"border": COLORS["border"], "primary": COLORS["green"],
                         "background": COLORS["bg"]}, children=[
            dcc.Tab(label="📅 Grid",      value="grid",     style={"color": COLORS["muted"]},
                    selected_style={"color": COLORS["green"], "background": COLORS["bg"]}),
            dcc.Tab(label="🧠 ML Insights", value="insights", style={"color": COLORS["muted"]},
                    selected_style={"color": COLORS["green"], "background": COLORS["bg"]}),
            dcc.Tab(label="📊 Charts",    value="charts",   style={"color": COLORS["muted"]},
                    selected_style={"color": COLORS["green"], "background": COLORS["bg"]}),
        ]),
        html.Div(id="tab-content", style={"padding": "24px 0"}),
    ]),
])


# ── CELL 7: Callbacks ─────────────────────────────────────────

# Add habit
@app.callback(
    Output("store-habits", "data"),
    Input("btn-add-habit", "n_clicks"),
    State("input-habit", "value"),
    State("store-habits", "data"),
    prevent_initial_call=True,
)
def add_habit(n, new_habit, habits):
    if new_habit and new_habit.strip() and new_habit.strip() not in habits:
        return habits + [new_habit.strip()]
    return habits


# Render tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("store-data", "data"),
    Input("store-habits", "data"),
    Input("dd-month", "value"),
    Input("dd-year", "value"),
)
def render_tab(tab, data, habits, month, year):
    df = run_ml_model(habits, data, year, month)
    days_in_month = calendar.monthrange(year, month)[1]
    today = date.today()
    is_current = (today.year == year and today.month == month)

    overall_score = int(df["ml_score"].mean()) if len(df) else 0
    overall_cons  = round(df["consistency"].mean(), 1) if len(df) else 0
    best_streak   = int(df["streak"].max()) if len(df) else 0

    # ── GRID TAB ──
    if tab == "grid":
        header = [html.Th("HABIT", style={**LABEL, "minWidth": "200px", "position": "sticky", "left": 0,
                                           "background": COLORS["card"], "zIndex": 2, "padding": "8px 12px"})]
        for d in range(1, days_in_month+1):
            is_today = is_current and d == today.day
            header.append(html.Th(str(d), style={
                "padding": "4px 2px", "textAlign": "center", "fontSize": "10px",
                "color": COLORS["green"] if is_today else COLORS["muted"],
                "minWidth": "26px", "fontWeight": "700" if is_today else "400",
            }))
        header.append(html.Th("%", style={**LABEL, "padding": "4px 8px"}))

        rows = []
        for _, row in df.iterrows():
            habit = row["habit"]
            row_data = data.get(habit, {})
            cells = [html.Td(habit, style={
                "padding": "5px 12px", "fontSize": "11px", "whiteSpace": "nowrap",
                "color": COLORS["text"], "position": "sticky", "left": 0,
                "background": COLORS["card"], "zIndex": 1,
                "maxWidth": "200px", "overflow": "hidden", "textOverflow": "ellipsis",
            })]
            for d in range(1, days_in_month+1):
                done = row_data.get(str(d), False)
                is_future = is_current and d > today.day
                cells.append(html.Td(
                    html.Div("✓" if done else "",
                        id={"type": "cell", "habit": habit, "day": d},
                        n_clicks=0,
                        style={
                            "width": "20px", "height": "20px", "borderRadius": "4px", "margin": "auto",
                            "background": COLORS["green"] if done else ("#0d0d1a" if is_future else COLORS["border"]),
                            "border": f"1px solid {'#00ff8788' if done else COLORS['border']}",
                            "cursor": "default" if is_future else "pointer",
                            "display": "flex", "alignItems": "center", "justifyContent": "center",
                            "fontSize": "10px", "color": "#0a0a14", "fontWeight": "700",
                            "transition": "all 0.15s",
                        }),
                    style={"padding": "3px", "textAlign": "center"},
                ))
            cons = row["consistency"]
            color = COLORS["green"] if cons >= 80 else COLORS["yellow"] if cons >= 50 else COLORS["red"]
            cells.append(html.Td(f"{cons}%", style={"padding": "4px 8px", "fontSize": "11px",
                                                      "fontWeight": "700", "color": color}))
            rows.append(html.Tr(cells, style={"borderBottom": f"1px solid #111"}))

        return html.Div([
            # Stats row
            dbc.Row([
                dbc.Col(stat_card("ML SCORE", f"{overall_score}/100",
                    COLORS["green"] if overall_score>=75 else COLORS["yellow"] if overall_score>=50 else COLORS["red"]), width=3),
                dbc.Col(stat_card("AVG CONSISTENCY", f"{overall_cons}%", COLORS["blue"]), width=3),
                dbc.Col(stat_card("BEST STREAK", f"{best_streak} days", COLORS["yellow"]), width=3),
                dbc.Col(stat_card("HABITS", str(len(habits)), COLORS["purple"]), width=3),
            ], className="mb-3"),

            # Grid
            html.Div(style={**CARD, "overflowX": "auto", "padding": "12px"}, children=[
                html.Table(
                    [html.Thead(html.Tr(header)), html.Tbody(rows)],
                    style={"borderCollapse": "collapse", "width": "max-content", "minWidth": "100%"},
                )
            ]),
            html.Div("Click any cell to toggle ✓ / ☐   |   Green = today's column   |   Dark = future days",
                     style={"fontSize": "11px", "color": COLORS["muted"], "textAlign": "center"}),
        ])

    # ── INSIGHTS TAB ──
    elif tab == "insights":
        cards = []
        for _, row in df.sort_values("ml_score", ascending=False).iterrows():
            insight = get_insight(row)
            score = row["ml_score"]
            color = COLORS["green"] if score>=75 else COLORS["yellow"] if score>=50 else COLORS["red"]
            cards.append(
                dbc.Col(html.Div(style={**CARD, "marginBottom": "12px"}, children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Div(f"{int(score)}", style={**BIG, "color": color, "fontSize": "32px"}),
                            html.Div("ML SCORE", style=LABEL),
                        ], style={"textAlign": "center"}), width=2),
                        dbc.Col([
                            html.Div(row["habit"], style={"fontSize": "12px", "color": COLORS["text"],
                                                           "fontWeight": "600", "marginBottom": "4px"}),
                            html.Div(insight, style={"fontSize": "11px", "color": color}),
                            dbc.Progress(value=score, style={"height": "4px", "marginTop": "8px"},
                                        color="success" if score>=75 else "warning" if score>=50 else "danger"),
                        ], width=6),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([html.Div(f"{int(row['consistency'])}%", style={"color": COLORS["blue"], "fontWeight":"700"}),
                                         html.Div("Consistency", style={**LABEL, "fontSize":"9px"})]),
                                dbc.Col([html.Div(f"{int(row['streak'])}d", style={"color": COLORS["yellow"], "fontWeight":"700"}),
                                         html.Div("Streak", style={**LABEL, "fontSize":"9px"})]),
                                dbc.Col([html.Div(f"{'+' if row['momentum']>0 else ''}{int(row['momentum'])}",
                                                  style={"color": COLORS["green"] if row['momentum']>0 else COLORS["red"], "fontWeight":"700"}),
                                         html.Div("Momentum", style={**LABEL, "fontSize":"9px"})]),
                                dbc.Col([html.Div(f"{row['predicted_rate']}%", style={"color": COLORS["purple"], "fontWeight":"700"}),
                                         html.Div("Forecast", style={**LABEL, "fontSize":"9px"})]),
                            ]),
                            dbc.Row([
                                dbc.Col(html.Div(f"Best: {row['best_day']}  Worst: {row['worst_day']}",
                                                 style={"fontSize":"10px","color":COLORS["muted"],"marginTop":"6px"})),
                            ])
                        ], width=4),
                    ], align="center"),
                ]), width=12)
            )

        return html.Div([
            dbc.Row([
                dbc.Col(stat_card("OVERALL ML SCORE", f"{overall_score}/100",
                    COLORS["green"] if overall_score>=75 else COLORS["yellow"] if overall_score>=50 else COLORS["red"]), width=3),
                dbc.Col(stat_card("AVG CONSISTENCY", f"{overall_cons}%", COLORS["blue"]), width=3),
                dbc.Col(stat_card("TOP STREAK", f"{best_streak} days", COLORS["yellow"]), width=3),
                dbc.Col(stat_card("TOTAL HABITS", str(len(habits)), COLORS["purple"]), width=3),
            ], className="mb-3"),
            dbc.Row(cards),
        ])

    # ── CHARTS TAB ──
    elif tab == "charts":
        return html.Div([
            dbc.Row([
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=score_gauge(overall_score), config={"displayModeBar": False})
                ]), width=3),
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=radar_chart(df), config={"displayModeBar": False})
                ]), width=5),
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=momentum_chart(df), config={"displayModeBar": False})
                ]), width=4),
            ]),
            dbc.Row([
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=heatmap_chart(df, data, year, month), config={"displayModeBar": False})
                ]), width=7),
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=bar_consistency(df), config={"displayModeBar": False})
                ]), width=5),
            ]),
            dbc.Row([
                dbc.Col(html.Div(style=CARD, children=[
                    dcc.Graph(figure=prediction_chart(df), config={"displayModeBar": False})
                ]), width=12),
            ]),
        ])


# Toggle habit cell
@app.callback(
    Output("store-data", "data"),
    Input({"type": "cell", "habit": ALL, "day": ALL}, "n_clicks"),
    State("store-data", "data"),
    State("dd-month", "value"),
    State("dd-year", "value"),
    prevent_initial_call=True,
)
def toggle_cell(n_clicks_list, data, month, year):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return data
    # only toggle on actual clicks
    if not any(n_clicks_list):
        return data
    habit = triggered["habit"]
    day   = str(triggered["day"])
    today = date.today()
    is_current = (today.year == year and today.month == month)
    if is_current and int(day) > today.day:
        return data
    current = data.get(habit, {})
    current[day] = not current.get(day, False)
    return {**data, habit: current}


# ── CELL 8: Launch ────────────────────────────────────────────

# If you have an ngrok auth token, set it here:
# ngrok.set_auth_token("YOUR_TOKEN_HERE")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
