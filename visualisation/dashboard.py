import polars as pl
import altair as alt
import numpy as np

# -----------------------------
# CONFIG & STYLE
# -----------------------------
COLORS = {
    "utilization": "#42a5f5",
    "queue": "#ec407a",
    "revenue": "#66bb6a",
}

COLUMN_WIDTH = 300 
GRID_SPACING = 40


# -----------------------------
# Fake data
# -----------------------------
def make_data(n=500):
    return pl.DataFrame({
        "utilization": np.clip(np.random.normal(0.5, 0.2, n), 0, 1),
        "queue": np.random.gamma(2, 2, n),
        "revenue": np.random.normal(200, 80, n).clip(0),
        "time": np.arange(n)
    }).to_pandas()


# -----------------------------
# Modular Components
# -----------------------------

def kpi_card(value: float, title: str, color: str):
    base = pl.DataFrame({"value": [value]}).to_pandas()
    
    background = alt.Chart(base).mark_rect(
        color="#1e1e1e", stroke="#444", strokeWidth=1.5, cornerRadius=6
    ).properties(width=COLUMN_WIDTH, height=100)

    text_base = alt.Chart(base).encode(x=alt.value(COLUMN_WIDTH / 2))

    value_text = text_base.mark_text(
        fontSize=32, fontWeight="bold", color=color, baseline="middle"
    ).encode(text=alt.Text("value:Q", format=".2f"), y=alt.value(60))

    title_text = text_base.mark_text(
        fontSize=14, color=color, opacity=0.7, baseline="middle"
    ).encode(text=alt.value(title), y=alt.value(25))

    return alt.layer(background, title_text, value_text)

def histogram(data, field, title):
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.85, color=COLORS[field])
        .encode(
            alt.X(f"{field}:Q", bin=alt.Bin(maxbins=30), title=None),
            alt.Y("count()", title=None)
        )
        .properties(title=title, width=COLUMN_WIDTH, height=180)
    )

def trend_line(data, field, title):
    return (
        alt.Chart(data)
        .mark_line(color=COLORS[field])
        .encode(
            alt.X("time:Q", title=None),
            alt.Y(f"{field}:Q", title=None)
        )
        .properties(title=title, width=COLUMN_WIDTH, height=180)
    )

def heatmap(data):
    full_width = (COLUMN_WIDTH * 3) + (GRID_SPACING * 2)
    
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("utilization:Q", bin=alt.Bin(maxbins=40), title="Utilization"),
            y=alt.Y("queue:Q", bin=alt.Bin(maxbins=40), title="Queue Size"),
            color=alt.Color(
                "count()",
                scale=alt.Scale(scheme="plasma"), # Other color schemes can be found at https://vega.github.io/vega/docs/schemes/
                title="Density",
                legend=alt.Legend(
                    orient="bottom", 
                    direction="horizontal", 
                    gradientLength=full_width * 0.35,
                    title="Concentration of events",
                    titleColor="#e0e0e0",
                    labelColor="#bbbbbb",
                    titleFontSize=14,
                    titlePadding=10
                )
            )
        )
        .properties(
            title="Density Heatmap",
            width=full_width,
            height=350
        )
    )

# -----------------------------
# Dashboard Builder
# -----------------------------
def build_dashboard():
    data = make_data()

    row_kpis = alt.hconcat(
        kpi_card(data["utilization"].mean(), "Avg Utilization", COLORS["utilization"]),
        kpi_card(data["queue"].mean(), "Avg Queue Size", COLORS["queue"]),
        kpi_card(data["revenue"].mean(), "Avg Revenue", COLORS["revenue"]),
        spacing=GRID_SPACING
    )

    row_dists = alt.hconcat(
        histogram(data, "utilization", "Utilization Distribution"),
        histogram(data, "queue", "Queue Distribution"),
        histogram(data, "revenue", "Revenue Distribution"),
        spacing=GRID_SPACING
    )

    row_trends = alt.hconcat(
        trend_line(data, "utilization", "Utilization Over Time"),
        trend_line(data, "queue", "Queue Over Time"),
        trend_line(data, "revenue", "Revenue Over Time"),
        spacing=GRID_SPACING
    )

    return (
        alt.vconcat(
            row_kpis,
            row_dists,
            row_trends,
            heatmap(data),
            spacing=50,
            center=True 
        )
        .resolve_legend(color='independent')
        .properties(
            title=alt.TitleParams(
                "Operational Dashboard",
                anchor="middle",
                fontSize=32,
                color="#bbbbbb",
                dy=-20
            )
        )
        .configure(background="#121212")
        .configure_view(stroke=None)
        .configure_axis(labelColor="#e0e0e0", titleColor="#e0e0e0", gridColor="#323232")
        .configure_title(color="white", fontSize=16)
    )

if __name__ == "__main__":
    chart = build_dashboard()
    chart.save("dashboard.svg")
    print("Dashboard generated successfully.")