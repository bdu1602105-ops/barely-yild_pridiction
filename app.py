import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Barley Grain Yield Predictor", page_icon="🌾")

MODEL_PATH = "barley_rf_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


pipeline = load_model()

# Pull the categories the model was actually trained on, so the dropdowns
# always match what the OneHotEncoder expects.
preprocessor = pipeline.named_steps["preprocessor"]
cat_transformer = preprocessor.named_transformers_["cat"]
categorical_cols = ["Nutrient", "Location", "Agro_Ecological_Zone", "SOIL_TYPE"]
cat_options = dict(zip(categorical_cols, cat_transformer.categories_))

st.title("🌾 Barley Grain Yield (GY) Predictor")
st.write(
    "Enter the trial conditions below to predict Grain Yield (GY) "
    "using the trained Random Forest model."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        nutrient = st.selectbox("Nutrient", options=list(cat_options["Nutrient"]))
        location = st.selectbox("Location", options=list(cat_options["Location"]))
        agro_zone = st.selectbox(
            "Agro-Ecological Zone", options=list(cat_options["Agro_Ecological_Zone"])
        )
        soil_type = st.selectbox("Soil Type", options=list(cat_options["SOIL_TYPE"]))

    with col2:
        rep = st.number_input("Rep", min_value=1, max_value=10, value=1, step=1)
        rate = st.number_input("Nutrient Rate", min_value=0.0, value=0.0, step=1.0)
        by = st.number_input("Biomass Yield (BY)", min_value=0.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict GY")

if submitted:
    input_df = pd.DataFrame(
        [{
            "Nutrient": nutrient,
            "Location": location,
            "Agro_Ecological_Zone": agro_zone,
            "SOIL_TYPE": soil_type,
            "Rep": rep,
            "Rate": rate,
            "BY": by,
        }]
    )

    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted Grain Yield (GY): **{prediction:.2f}**")

    with st.expander("View input data"):
        st.dataframe(input_df)

st.markdown("---")
st.caption(
    "Model: Random Forest Regressor trained on barley trial data "
    "(Nutrient, Location, Agro-Ecological Zone, Soil Type, Rep, Rate, BY → GY)."
)
