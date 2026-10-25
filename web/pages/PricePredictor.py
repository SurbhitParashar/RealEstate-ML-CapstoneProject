import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Viz Demo")



with open('df.pkl','rb') as file:
    df = pickle.load(file)


with open('pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

WEB_DIR=Path(__file__).resolve().parents[1]
MODEL_PATH=WEB_DIR / "pipeline.pkl"
DATA_PATH=WEB_DIR / "df.pkl"

@st.cache_resource
def load_model_and_explainer(model_path,model_version):
    with open(model_path, "rb") as file:
        fitted_pipeline=pickle.load(file)

    forest=fitted_pipeline.steps[-1][1]
    tree_explainer=shap.TreeExplainer(
        forest,
        feature_perturbation="tree_path_dependent",
        model_output="raw"    
    )
    return fitted_pipeline, tree_explainer

@st.cache_data
def load_reference_data(data_path,data_version):
    with open(data_path,"rb") as file:
        return pickle.load(file)

pipeline,explainer=load_model_and_explainer(str(MODEL_PATH),MODEL_PATH.stat().st_mtime_ns)

df=load_reference_data(str(DATA_PATH),DATA_PATH.stat().st_mtime_ns)


st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))
facing = st.selectbox('Facing',sorted(df['facing'].unique().tolist()))
built_up_area = float(st.number_input('Built Up Area'))
servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
property_age = st.selectbox('Property Age',sorted(df['age_category'].unique().tolist()))


furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
rating=float(st.selectbox('Rating',sorted(df['combined_rating'].unique().tolist())))

luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))

floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

# property_type	sector	bedRoom	bathroom	balcony	facing	built_up_area	servant room	age_category	furnishing_type	combined_rating	luxury_category	floor_category
# 0	house	sohna road	4	3	3+	East	2750	0	New Property	Unfurnished	4	Low	Low Floor

FEATURE_LABELS = {
    "property_type": "Property type",
    "sector": "Sector",
    "bedRoom": "Bedrooms",
    "bathroom": "Bathrooms",
    "balcony": "Balconies",
    "facing": "Facing",
    "built_up_area": "Built-up area",
    "servant room": "Servant room",
    "age_category": "Property age",
    "furnishing_type": "Furnishing",
    "combined_rating": "Rating",
    "luxury_category": "Luxury category",
    "floor_category": "Floor category",
}


def explain_prediction(input_df, predicted_log_price):
    preprocessor = pipeline[:-1]

    # Use the existing fitted preprocessing.
    transformed_input = preprocessor.transform(input_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    transformed_input = np.asarray(
        transformed_input,
        dtype=float,
    )

    transformed_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    explanation = explainer(
        transformed_input,
        check_additivity=True,
    )

    shap_values = np.asarray(
        explanation.values[0],
        dtype=float,
    ).reshape(-1)

    reference_log_price = float(
        np.asarray(explanation.base_values).reshape(-1)[0]
    )

    if len(transformed_names) != len(shap_values):
        raise ValueError(
            "Transformed feature names and SHAP values do not match."
        )

    # Group transformed columns back into the original inputs.
    # For example, all age-category dummy columns become Property age.
    contributions = {
        column: 0.0 for column in input_df.columns
    }

    original_columns = sorted(
        input_df.columns,
        key=len,
        reverse=True,
    )

    for transformed_name, contribution in zip(
        transformed_names,
        shap_values,
    ):
        # Example: "num__built_up_area" -> "built_up_area"
        feature_name = transformed_name.split("__", 1)[-1]

        original_feature = next(
            (
                column
                for column in original_columns
                if feature_name == column
                or feature_name.startswith(column + "_")
            ),
            None,
        )

        if original_feature is None:
            raise ValueError(
                f"Cannot map transformed feature: {transformed_name}"
            )

        contributions[original_feature] += float(contribution)

    grouped_values = np.array(
        [contributions[column] for column in input_df.columns],
        dtype=float,
    )

    # Verify that the grouped explanation reconstructs the model output.
    reconstructed_log_price = (
        reference_log_price + grouped_values.sum()
    )

    if not np.isclose(
        reconstructed_log_price,
        predicted_log_price,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "SHAP contributions do not reconstruct the prediction."
        )

    st.subheader("Why this price?")

    st.caption(
        "Positive contributions push this prediction above the "
        "model reference; negative contributions push it below. "
        "Contributions are in log-price units, not crores or lakhs."
    )

    # Include original values in the chart labels.
    chart_labels = [
        (
            f"{FEATURE_LABELS.get(column, column)}"
            f" = {input_df.iloc[0][column]}"
        )
        for column in input_df.columns
    ]

    grouped_explanation = shap.Explanation(
        values=grouped_values,
        base_values=reference_log_price,
        feature_names=chart_labels,
    )

    shap.plots.waterfall(
        grouped_explanation,
        max_display=len(input_df.columns),
        show=False,
    )

    fig = plt.gcf()

    try:
        fig.set_size_inches(11, 8)
        fig.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)

    # Build a table sorted by contribution magnitude.
    rows = []

    for column, contribution in contributions.items():
        if contribution > 1e-8:
            direction = "Pushes up"
        elif contribution < -1e-8:
            direction = "Pushes down"
        else:
            direction = "Negligible"

        rows.append(
            {
                "Feature": FEATURE_LABELS.get(column, column),
                "Entered value": str(input_df.iloc[0][column]),
                "SHAP contribution (log-price)": contribution,
                "Direction": direction,
            }
        )

    table = pd.DataFrame(rows)

    order = (
        table["SHAP contribution (log-price)"]
        .abs()
        .sort_values(ascending=False)
        .index
    )

    table = table.loc[order].reset_index(drop=True)

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "SHAP contribution (log-price)": st.column_config.NumberColumn(
                "SHAP contribution (log-price)",
                format="%.4f",
            )
        },
    )

    positive = [
        column
        for column, value in contributions.items()
        if value > 1e-8
    ]

    negative = [
        column
        for column, value in contributions.items()
        if value < -1e-8
    ]

    if positive:
        strongest_up = max(
            positive,
            key=contributions.get,
        )
        st.write(
            "**Strongest upward factor:** "
            + FEATURE_LABELS.get(strongest_up, strongest_up)
        )

    if negative:
        strongest_down = min(
            negative,
            key=contributions.get,
        )
        st.write(
            "**Strongest downward factor:** "
            + FEATURE_LABELS.get(strongest_down, strongest_down)
        )

    st.caption(
        "These contributions explain this model prediction relative "
        "to its reference. They do not establish causal price changes."
    )



if st.button('Predict'):

    
    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, facing, built_up_area, servant_room,  property_age, furnishing_type, rating,luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'facing',
                'built_up_area', 'servant room', 'age_category',
               'furnishing_type', 'combined_rating','luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    if built_up_area <= 0:
        st.error("Enter a built-up area greater than zero.")
        st.stop()

    

    # predict log-price, then convert to crores
    predicted_log_price=float(pipeline.predict(one_df)[0])
    base_price = float(np.expm1(predicted_log_price))
    low = base_price - 0.22
    high = base_price + 0.22

    st.metric(
        "Predicted price",
        f"₹{base_price:.2f} Cr"
    )

    st.write(
        f"Existing estimated range: "
        f"₹{low:.2f} Cr to ₹{high:.2f} Cr"
    )

    with st.spinner("Calculating feature contribution..."):
        try:
            explain_prediction(one_df, predicted_log_price)
        except Exception as e:
            st.error("The price was calculated, but its explanation "
                "could not be generated.")

            st.exception(e)
   