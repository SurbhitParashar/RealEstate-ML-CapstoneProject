from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Model diagnostics", layout="wide")
st.title("Model diagnostics")
st.write("Explore where property price predictions succeed and where they miss.")
st.caption(
    "Historical test-set evaluation from the exported notebook results. "
    "Prices are in crores (1 Cr = 100 lakh). Residual = actual − predicted: "
    "positive means underprediction; negative means overprediction."
)

DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "model_diagnostics.csv"
GROUPS = {
    "Property type": "property_type",
    "Sector": "sector",
    "Bedrooms": "bedRoom",
    "Property age": "age_category",
    "Furnishing": "furnishing_type",
    "Luxury category": "luxury_category",
    "Actual price band": "price_band",
    "Area band": "area_band",
}


@st.cache_data(max_entries=2)
def load_diagnostics(path, version):
    data = pd.read_csv(path)
    required = {"actual_price_cr", "predicted_price_cr", "built_up_area", "property_type", "sector", "bedRoom"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    for column in ["actual_price_cr", "predicted_price_cr", "built_up_area"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.empty:
        raise ValueError("The exported evaluation file has no rows.")
    if not np.isfinite(data[["actual_price_cr", "predicted_price_cr"]].to_numpy()).all():
        raise ValueError("Actual and predicted prices must contain finite numbers in crores.")

    actual = data["actual_price_cr"]
    data["residual_cr"] = actual - data["predicted_price_cr"]
    data["absolute_error_cr"] = data["residual_cr"].abs()
    data["squared_error_cr2"] = data["residual_cr"] ** 2
    data["absolute_error_lakh"] = data["absolute_error_cr"] * 100
    data["percentage_error"] = data["residual_cr"] / actual.where(actual > 0) * 100
    data["absolute_percentage_error"] = data["percentage_error"].abs()
    data["error_direction"] = np.select(
        [data["residual_cr"] > 0, data["residual_cr"] < 0],
        ["Underpredicted", "Overpredicted"], default="Exact",
    )
    data["price_band"] = pd.cut(
        actual, [0, 1, 2, 5, 10, np.inf],
        labels=["≤1 Cr", "1–2 Cr", "2–5 Cr", "5–10 Cr", ">10 Cr"],
        include_lowest=True,
    )
    data["area_band"] = pd.cut(
        data["built_up_area"], [0, 1000, 2000, 3000, 5000, np.inf],
        labels=["≤1000 sqft", "1000–2000 sqft", "2000–3000 sqft", "3000–5000 sqft", ">5000 sqft"],
        include_lowest=True,
    )
    return data


def summarize_errors(data, group):
    summary = data.groupby(group, observed=True, dropna=False).agg(
        Count=("residual_cr", "size"),
        MAE=("absolute_error_cr", "mean"),
        RMSE=("squared_error_cr2", lambda values: np.sqrt(values.mean())),
        Bias=("residual_cr", "mean"),
        Median_percentage_error=("absolute_percentage_error", "median"),
    ).reset_index()
    summary["Small sample"] = summary["Count"] < 10
    return summary.sort_values("MAE", ascending=False)


if not DATA_PATH.exists():
    st.info("Export your notebook's test-set diagnostics to web/datasets/model_diagnostics.csv first.")
    st.stop()
try:
    diagnostics = load_diagnostics(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
except (ValueError, OSError) as error:
    st.error(f"Cannot load evaluation results: {error}")
    st.stop()

with st.sidebar:
    st.header("Evaluation filters")
    property_types = st.multiselect(
        "Property type", sorted(diagnostics["property_type"].dropna().unique()),
        key="diagnostics_types", help="Leave empty to include all types.",
    )
    sectors = st.multiselect(
        "Sector", sorted(diagnostics["sector"].dropna().unique()),
        key="diagnostics_sectors", help="Leave empty to include all sectors.",
    )
    price_bands = st.multiselect(
        "Actual price band", list(diagnostics["price_band"].cat.categories),
        key="diagnostics_prices", help="Leave empty to include all prices.",
    )
    area_bands = st.multiselect(
        "Area band", list(diagnostics["area_band"].cat.categories),
        key="diagnostics_areas", help="Leave empty to include all areas.",
    )
    st.caption("Actual-price bands describe known evaluation properties, not future inputs.")

filtered = diagnostics.copy()
for column, selected in [
    ("property_type", property_types), ("sector", sectors),
    ("price_band", price_bands), ("area_band", area_bands),
]:
    if selected:
        filtered = filtered.loc[filtered[column].isin(selected)]

st.caption(f"Showing {len(filtered):,} of {len(diagnostics):,} evaluated properties. All results below follow the filters.")
if filtered.empty:
    st.info("No evaluated properties match these filters. Clear a filter to continue.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Properties", f"{len(filtered):,}", border=True)
    st.metric("MAE", f"₹{filtered['absolute_error_cr'].mean() * 100:.2f} lakh", border=True)
    st.metric("RMSE", f"₹{np.sqrt(filtered['squared_error_cr2'].mean()) * 100:.2f} lakh", border=True)
    st.metric("Mean residual", f"₹{filtered['residual_cr'].mean() * 100:+.2f} lakh", border=True)

percentage_errors = filtered["absolute_percentage_error"].dropna()
with st.container(horizontal=True):
    for label, value in [
        ("Median absolute error", f"₹{filtered['absolute_error_cr'].median() * 100:.2f} lakh"),
        ("90th percentile error", f"₹{filtered['absolute_error_cr'].quantile(.9) * 100:.2f} lakh"),
        ("Within 10% of actual", f"{percentage_errors.le(10).mean() * 100:.1f}%" if len(percentage_errors) else "N/A"),
        ("Within 20% of actual", f"{percentage_errors.le(20).mean() * 100:.1f}%" if len(percentage_errors) else "N/A"),
    ]:
        st.metric(label, value, border=True)
st.caption(
    f"Underpredicted: {filtered['residual_cr'].gt(0).mean():.1%}. "
    f"Percentage metrics use {len(percentage_errors):,} properties with positive actual prices. "
    "The 90th percentile is an observed error threshold, not a prediction interval."
)

st.subheader("Errors by segment")
available_groups = {label: column for label, column in GROUPS.items() if column in filtered}
group_label = st.selectbox("Compare by", list(available_groups))
group = available_groups[group_label]
summary = summarize_errors(filtered, group)
st.caption("Amounts below are in Cr. Positive bias means underprediction. Groups below 10 properties are flagged; this is a display rule, not a significance test.")
st.dataframe(summary, hide_index=True, column_config={
    "MAE": st.column_config.NumberColumn("MAE (Cr)", format="%.3f"),
    "RMSE": st.column_config.NumberColumn("RMSE (Cr)", format="%.3f"),
    "Bias": st.column_config.NumberColumn("Mean residual (Cr)", format="%.3f"),
    "Median_percentage_error": st.column_config.NumberColumn("Median absolute error (%)", format="%.1f%%"),
})
chart_summary = summary.head(20).copy()
chart_summary[group] = chart_summary[group].astype(str)
st.altair_chart(alt.Chart(chart_summary).mark_bar().encode(
    x=alt.X(f"{group}:N", sort="-y", title=group_label),
    y=alt.Y("MAE:Q", title="MAE (Cr)"),
    tooltip=[group, "Count", "MAE", "RMSE", "Bias"],
))
if len(summary) > 20:
    st.caption("Chart shows the 20 segments with highest MAE; the table includes all segments.")

st.subheader("Residual distribution")
zero = alt.Chart(pd.DataFrame({"zero": [0]}))
histogram = alt.Chart(filtered).mark_bar().encode(
    x=alt.X("residual_cr:Q", bin=alt.Bin(maxbins=35), title="Actual − predicted price (Cr)"),
    y=alt.Y("count():Q", title="Properties"), tooltip=[alt.Tooltip("count():Q", title="Properties")],
)
st.altair_chart(histogram + zero.mark_rule(color="red", strokeDash=[5, 5]).encode(x="zero:Q"))

st.subheader("Systematic error patterns")
pattern = st.selectbox("Compare residuals against", ["Predicted price", "Built-up area"])
x_column, x_title = (
    ("predicted_price_cr", "Predicted price (Cr)") if pattern == "Predicted price"
    else ("built_up_area", "Built-up area (sq ft)")
)
scatter = alt.Chart(filtered).mark_circle(size=45, opacity=.6).encode(
    x=alt.X(f"{x_column}:Q", title=x_title),
    y=alt.Y("residual_cr:Q", title="Actual − predicted price (Cr)"),
    color=alt.Color("property_type:N", title="Property type"),
    tooltip=["sector", "property_type", "built_up_area", "actual_price_cr", "predicted_price_cr", "residual_cr"],
)
st.altair_chart(scatter + zero.mark_rule(color="red", strokeDash=[5, 5]).encode(y="zero:Q"))
st.caption("Above zero: underprediction. Below zero: overprediction. Trends and widening spread identify patterns to investigate; they do not establish a cause.")

st.subheader("Worst predictions")
ranking = st.selectbox("Rank by", ["Absolute error (Cr)", "Absolute percentage error"])
limit = st.slider("Properties to display", min_value=5, max_value=50, value=20, step=5)
sort_column = "absolute_error_cr" if ranking == "Absolute error (Cr)" else "absolute_percentage_error"
worst = filtered.dropna(subset=[sort_column]).nlargest(limit, sort_column)
columns = [column for column in [
    "source_row_id", "property_type", "sector", "bedRoom", "built_up_area",
    "actual_price_cr", "predicted_price_cr", "residual_cr", "absolute_error_lakh",
    "absolute_percentage_error", "error_direction",
] if column in worst]
st.dataframe(worst[columns], hide_index=True, column_config={
    column: st.column_config.NumberColumn(format="%.2f") for column in columns
    if column.endswith("_cr") or column in ["absolute_error_lakh", "absolute_percentage_error"]
})
st.download_button("Download filtered diagnostics", filtered.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_model_diagnostics.csv", mime="text/csv")
st.download_button("Download segment summary", summary.to_csv(index=False).encode("utf-8"),
                   file_name="model_segment_summary.csv", mime="text/csv")
