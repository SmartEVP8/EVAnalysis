import polars as pl
import altair as alt
import webbrowser
import os

def main():
    data_path = "userdata.parquet"
    df = pl.read_parquet(data_path).select("salary").drop_nulls()

    chart = alt.Chart(df).mark_bar().encode(
        alt.X("salary:Q", bin=alt.Bin(maxbins=50), title="Salary"),
        alt.Y("count()", title="Count"),
        tooltip=[
            alt.Tooltip("salary:Q", bin=alt.Bin(maxbins=50), title="Salary Range"),
            alt.Tooltip("count()", title="Count"),
        ]
    ).properties(
        title="Salary Distribution",
        width=700,
        height=400,
    ).interactive()

    html_path = os.path.abspath("chart.html")
    chart.save(html_path)
    webbrowser.open(f"file://{html_path}")

if __name__ == "__main__":
    main()
