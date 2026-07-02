import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recommend Apartments", layout="wide")

# ---------------- LOAD DATA ---------------- #

location_df = pickle.load(open('datasets/location_df.pkl','rb'))

cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl','rb'))
cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl','rb'))
cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl','rb'))


# ------------------------------------------------------ #
# Recommendation Function
# ------------------------------------------------------ #

def recommend_properties_with_scores(property_name, top_n=5):

    cosine_sim_matrix = (
        0.5 * cosine_sim1 +
        0.8 * cosine_sim2 +
        1.0 * cosine_sim3
    )

    idx = location_df.index.get_loc(property_name)

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_scores = [i[1] for i in sim_scores[1:top_n+1]]

    recommendations = pd.DataFrame({
        "PropertyName": location_df.index[top_indices],
        "SimilarityScore": top_scores
    })

    return recommendations


# ------------------------------------------------------ #
# Session State
# ------------------------------------------------------ #

if "selected_property" not in st.session_state:
    st.session_state.selected_property = None

if "search_results" not in st.session_state:
    st.session_state.search_results = None


# ------------------------------------------------------ #
# SEARCH BY LOCATION
# ------------------------------------------------------ #

st.title("Search Apartments")

location_columns = [
    col for col in location_df.columns
    if col != "link"
]

selected_location = st.selectbox(
    "Choose Location",
    sorted(location_columns)
)

radius = st.number_input(
    "Radius (Km)",
    min_value=0.0,
    step=0.5
)

if st.button("Search"):

    st.session_state.selected_location = selected_location
    st.session_state.radius = radius

    st.session_state.search_results = (
        location_df[
            location_df[selected_location] < radius * 1000
        ][selected_location].sort_values()
    )

if st.session_state.search_results is not None:

    

    result_ser = st.session_state.search_results

    if len(result_ser) == 0:
        st.warning("No properties found.")

    else:
        st.subheader("Properties Found")
        for property_name, distance in result_ser.items():

            c1, c2, c3 = st.columns([5,2,3])

            with c1:
                st.write(
                    f"**{property_name}** ({round(distance/1000,2)} Km)"
                )

            with c2:
                st.link_button(
                    "Open Link",
                    location_df.loc[property_name, "link"]
                )

            with c3:
                if st.button(
                    "Recommend Similar",
                    key=f"rec_{property_name}"
                ):
                    st.session_state.selected_property = property_name


# ------------------------------------------------------ #
# SHOW RECOMMENDATIONS
# ------------------------------------------------------ #

if st.session_state.selected_property is not None:

    st.divider()

    st.header(
        f"Properties similar to\n\n{st.session_state.selected_property}"
    )

    recommendation_df = recommend_properties_with_scores(
        st.session_state.selected_property,
        top_n=5
    )

    for _, row in recommendation_df.iterrows():

        property_name = row["PropertyName"]
        score = row["SimilarityScore"]

        c1, c2 = st.columns([6,2])

        with c1:
            st.write(
                f"**{property_name}** | Similarity Score : {score:.3f}"
            )

        with c2:
            st.link_button(
                "Open Link",
                location_df.loc[property_name, "link"],
                key=f"link_{property_name}"
            )