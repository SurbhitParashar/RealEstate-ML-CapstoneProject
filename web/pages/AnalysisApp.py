import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from wordcloud import WordCloud

st.set_page_config(
    page_title="Property Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Minimal styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    h1 { font-size: 1.9rem !important; margin-bottom: 0 !important; }
    h2 { font-size: 1.25rem !important; color: #333; margin-top: 0.2rem; }
    .section-label {
        font-size: 0.78rem;
        color: #888;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fc;
        border: 1px solid #e4e6ee;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e4e6ee;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    new_df = pd.read_csv('./datasets/dataviz1.csv')
    df2    = pd.read_csv('./datasets/dataviz2.csv')

    def parse_features(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    df2["features"] = df2["features"].apply(parse_features)
    return new_df, df2

new_df, df = load_data()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🏠 Gurgaon Property Analytics")
st.caption("Exploring real estate trends across sectors — prices, amenities, and investment signals.")
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Listings",    f"{len(new_df):,}")
k2.metric("Avg Price",         f"₹ {new_df['price'].mean():.2f} Cr")
k3.metric("Avg Price / sqft",  f"₹ {new_df['price_per_sqft'].mean():,.0f}")
k4.metric("Sectors Covered",   new_df['sector'].nunique())

st.divider()

# ── 1. Sector Geomap ───────────────────────────────────────────────────────────
st.subheader("📍 Sector-wise Price per Sqft — Map")
st.caption("Bubble size = avg built-up area · Colour = price per sqft")

group_df = (
    new_df.groupby('sector')
          .mean(numeric_only=True)
          [['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']]
)

fig_map = px.scatter_map(
    group_df,
    lat="latitude", lon="longitude",
    color="price_per_sqft", size="built_up_area",
    color_continuous_scale=px.colors.cyclical.IceFire,
    zoom=10, map_style="open-street-map",
    hover_name=group_df.index
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ── 2. Investment Bubble Chart ─────────────────────────────────────────────────
st.subheader("💰 Sector Investment Overview")
st.caption("Luxury vs Price · Bubble size = avg rating · Hover for details")

sector_stats = (
    new_df.groupby("sector")
          .agg(
              avg_price        = ("price",          "mean"),
              avg_luxury_score = ("luxury_score",   "mean"),
              avg_rating       = ("combined_rating","mean"),
              properties       = ("sector",         "count")
          )
          .reset_index()
          .dropna(subset=["avg_price", "avg_luxury_score", "avg_rating"])
)

fig_invest = px.scatter(
    sector_stats,
    x="avg_luxury_score", y="avg_price",
    size="avg_rating", color="avg_price",
    hover_name="sector",
    hover_data={
        "avg_price":        ":.2f",
        "avg_luxury_score": ":.1f",
        "avg_rating":       ":.2f",
        "properties":       True
    },
    color_continuous_scale="Turbo",
    size_max=40,
    labels={
        "avg_luxury_score": "Avg Luxury Score",
        "avg_price":        "Avg Price (Cr)",
        "avg_rating":       "Avg Rating"
    }
)
fig_invest.update_layout(template="plotly_white")
st.plotly_chart(fig_invest, use_container_width=True)

st.divider()

# ── 3. Top Sectors + BHK Pie (side by side) ────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("🏆 Top 15 Most Expensive Sectors")
    sector_price = (
        new_df.groupby('sector')['price']
              .mean()
              .sort_values(ascending=True)
              .tail(15)
    )
    fig_bar = px.bar(
        sector_price,
        orientation='h',
        color=sector_price.values,
        color_continuous_scale="Blues",
        labels={"value": "Avg Price (Cr)", "index": "Sector"}
    )
    fig_bar.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        showlegend=False,
        yaxis_title="",
        xaxis_title="Avg Price (Cr)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("🛏 BHK Distribution")
    sector_opts = ["Overall"] + sorted(new_df['sector'].dropna().unique().tolist())
    sel_sector_pie = st.selectbox("Filter by Sector", sector_opts, key="pie_sec")

    pie_df = new_df if sel_sector_pie == "Overall" else new_df[new_df['sector'] == sel_sector_pie]
    fig_pie = px.pie(
        pie_df, names='bedRoom',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── 4. Area vs Price ───────────────────────────────────────────────────────────
st.subheader("📐 Area vs Price")
st.caption("Coloured by number of bedrooms")

prop_type = st.selectbox("Property Type", ["flat", "house"], key="area_prop")
fig_scatter = px.scatter(
    new_df[new_df['property_type'] == prop_type],
    x="built_up_area", y="price",
    color="bedRoom",
    labels={"built_up_area": "Built-up Area (sqft)", "price": "Price (Cr)"},
    color_continuous_scale="Viridis"
)
fig_scatter.update_layout(template="plotly_white")
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── 5. Box Plot + KDE dist (side by side) ──────────────────────────────────────
col3, col4 = st.columns(2, gap="large")

with col3:
    st.subheader("📦 BHK Price Range")
    fig_box = px.box(
        new_df[new_df['bedRoom'] <= 4],
        x='bedRoom', y='price',
        color='bedRoom',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"bedRoom": "Bedrooms", "price": "Price (Cr)"}
    )
    fig_box.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with col4:
    st.subheader("📊 Price Distribution — House vs Flat")
    fig_dist, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        new_df[new_df['property_type'] == 'house']['price'],
        label='House', fill=True, alpha=0.45, ax=ax
    )
    sns.kdeplot(
        new_df[new_df['property_type'] == 'flat']['price'],
        label='Flat', fill=True, alpha=0.45, ax=ax
    )
    ax.set_xlabel("Price (Cr)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_dist)

st.divider()

# ── 6. Amenities Word Cloud ────────────────────────────────────────────────────
st.subheader("☁️ Amenities Word Cloud")

sel_sector_wc = st.selectbox(
    "Select Sector",
    ["Overall"] + sorted(df["sector"].dropna().unique().tolist()),
    key="wc_sec"
)

temp_df = df if sel_sector_wc == "Overall" else df[df["sector"] == sel_sector_wc]

feature_counts = Counter()
for fl in temp_df["features"]:
    feature_counts.update(fl)

if feature_counts:
    wc = WordCloud(
        width=1400, height=500,
        background_color="white",
        collocations=False
    ).generate_from_frequencies(feature_counts)

    fig_wc, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig_wc)
else:
    st.info("No amenity data available for this sector.")

st.divider()

# ── 7. Correlation Heatmap (collapsed by default) ─────────────────────────────
with st.expander("🔗 Correlation Heatmap  (advanced)"):
    st.caption("Pairwise correlations between all numeric features")
    corr = new_df.select_dtypes(include='number').corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig_corr.update_layout(template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)