
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------
# 1. Page configuration
# ----------------------------------------

st.set_page_config(
    page_title="Planet Hunter | AI Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "planet_model.pkl"
FEATURE_PATH = BASE / "models" / "feature_names.pkl"
DATA_PATH = BASE / "data" / "kepler_cleaned.csv"
RESULTS_PATH = BASE / "reports" / "model_comparison.csv"
CV_PATH = BASE / "reports" / "cross_validation_results.csv"

# ----------------------------------------
# 2. Visual styling
# ----------------------------------------

st.markdown("""
<style>
.stApp {
    background: #08111f;
    color: #e8f1ff;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f0f6ff !important;
}

[data-testid="stMetric"] {
    background: #12233b;
    border: 1px solid #263c59;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stSidebar"] {
    background: #0d1b2e;
}

div.stButton > button {
    background: #2377df;
    color: white;
    border-radius: 9px;
    border: none;
}

div.stButton > button:hover {
    background: #398df2;
    color: white;
}

.subtitle {
    color: #a9bdd7;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 3. Load trained model and dataset
# ----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

try:
    model = load_model()
    df = load_data()
    results = load_results()
    features = joblib.load(FEATURE_PATH)

except Exception as error:
    st.error(f"Unable to load required project files: {error}")
    st.stop()

# ----------------------------------------
# 4. Helper functions
# ----------------------------------------

def predict_object(input_data):
    frame = pd.DataFrame(
        [input_data],
        columns=features
    )

    prediction = int(model.predict(frame)[0])

    probability = float(
        model.predict_proba(frame)[0, 1]
    )

    return prediction, probability


def class_name(value):
    if value == 1:
        return "PLANET CANDIDATE"
    return "FALSE POSITIVE"


def get_object_input(row):
    return {
        feature: row[feature]
        for feature in features
    }


def show_prediction(prediction, probability):
    if prediction == 1:
        st.success("Classification: Planet Candidate")
    else:
        st.error("Classification: False Positive")

    st.metric(
        "Candidate-class probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        "This is a model score, not independent "
        "confirmation of a planet. Probability "
        "calibration has not been verified."
    )


# ----------------------------------------
# 5. Sidebar navigation
# ----------------------------------------

with st.sidebar:
    st.title("PLANET HUNTER")
    st.caption("AI-Powered Exoplanet Detection")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Mission Control",
            "Planet Analyzer",
            "What-If Simulator",
            "Exoplanet Explorer"
        ]
    )

    st.divider()

    st.caption("Dataset: NASA Kepler KOI Catalog")
    st.caption("Model: Scikit-learn Pipeline")
    st.caption("Mode: Offline scientific demonstration")


# ----------------------------------------
# 6. Mission Control
# ----------------------------------------

if page == "Mission Control":

    st.title("MISSION CONTROL")
    st.markdown(
        '<p class="subtitle">Explore astronomical data '
        'through machine learning.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Objects in Dataset",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Candidate / Confirmed",
            f"{int((df['target'] == 1).sum()):,}"
        )

    with col3:
        st.metric(
            "False Positives",
            f"{int((df['target'] == 0).sum()):,}"
        )

    st.subheader("Kepler Object Classification")

    counts = df["target"].map({
        0: "False Positive",
        1: "Candidate / Confirmed"
    }).value_counts().reset_index()

    counts.columns = ["Category", "Count"]

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        hole=0.55,
        color="Category",
        color_discrete_map={
            "Candidate / Confirmed": "#37c99a",
            "False Positive": "#ef7373"
        }
    )

    fig.update_layout(
        paper_bgcolor="#08111f",
        font_color="#e8f1ff",
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Machine Learning Model Comparison")

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    if "F1_Score" in results.columns:
        fig2 = px.bar(
            results,
            x="Model",
            y="F1_Score",
            color="Model",
            title="Held-Out Test F1-Score"
        )

        fig2.update_layout(
            paper_bgcolor="#08111f",
            plot_bgcolor="#08111f",
            font_color="#e8f1ff",
            showlegend=False
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    if CV_PATH.exists():
        cv_results = pd.read_csv(CV_PATH)

        if len(cv_results) > 0:
            best_model_name = cv_results.iloc[
                cv_results["CV_F1_Mean"].argmax()
            ]["Model"]

            st.info(
                f"Selected model using training CV: "
                f"{best_model_name}"
            )

    st.subheader("Evaluation Reports")

    col_a, col_b = st.columns(2)

    with col_a:
        path = BASE / "reports" / "confusion_matrix.png"

        if path.exists():
            st.image(
                str(path),
                caption="Confusion Matrix"
            )

    with col_b:
        path = BASE / "reports" / "roc_curve.png"

        if path.exists():
            st.image(
                str(path),
                caption="ROC Curve"
            )


# ----------------------------------------
# 7. Planet Analyzer
# ----------------------------------------

elif page == "Planet Analyzer":

    st.title("PLANET ANALYZER")

    st.write(
        "Select a real Kepler catalog object "
        "and run the trained classifier."
    )

    available_ids = df.index.tolist()

    selected_id = st.selectbox(
        "Select dataset record",
        available_ids,
        format_func=lambda idx: (
            f"Record {idx} | "
            f"Star {df.loc[idx, 'kepid']}"
        )
    )

    row = df.loc[selected_id]

    input_data = get_object_input(row)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Orbital Period",
            f"{row['koi_period']:.2f} days"
        )

    with col2:
        st.metric(
            "Transit Depth",
            f"{row['koi_depth']:.2f} ppm"
        )

    with col3:
        st.metric(
            "Planetary Radius",
            f"{row['koi_prad']:.2f} Earth radii"
        )

    with st.expander("View astronomical measurements"):
        st.dataframe(
            pd.DataFrame(
                list(input_data.items()),
                columns=["Feature", "Value"]
            ),
            hide_index=True,
            use_container_width=True
        )

    if st.button(
        "Analyze Selected Object",
        type="primary"
    ):
        prediction, probability = predict_object(
            input_data
        )

        show_prediction(
            prediction,
            probability
        )

        actual_label = class_name(
            int(row["target"])
        )

        st.info(
            f"Recorded catalog category: {actual_label}"
        )

        st.caption(
            "Catalog records may include objects used "
            "in model training. This screen is for "
            "demonstration, not independent evaluation."
        )


# ----------------------------------------
# 8. What-If Simulator
# ----------------------------------------

elif page == "What-If Simulator":

    st.title("WHAT-IF PLANET SIMULATOR")

    st.write(
        "Choose a catalog object as a baseline. "
        "Modify three measurements and observe "
        "how the classification changes."
    )

    selected_id = st.selectbox(
        "Baseline record",
        df.index.tolist(),
        format_func=lambda idx: (
            f"Record {idx} | Star {df.loc[idx, 'kepid']}"
        )
    )

    baseline_row = df.loc[selected_id]
    baseline = get_object_input(baseline_row)

    input_data = baseline.copy()

    editable_features = {
        "koi_period": (
            "Orbital period (days)",
            0.01
        ),
        "koi_depth": (
            "Transit depth (ppm)",
            0.01
        ),
        "koi_duration": (
            "Transit duration (hours)",
            0.01
        )
    }

    st.subheader("Adjust Signal Properties")

    for feature, (label, minimum) in editable_features.items():

        raw_value = baseline[feature]

        if pd.isna(raw_value):
            raw_value = df[feature].median()

        value = max(
            float(raw_value),
            minimum
        )

        input_data[feature] = st.number_input(
            label,
            min_value=minimum,
            value=value,
            format="%.3f"
        )

    if st.button(
        "Run Simulation",
        type="primary"
    ):

        original_pred, original_prob = predict_object(
            baseline
        )

        new_pred, new_prob = predict_object(
            input_data
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Signal")

            st.metric(
                "Candidate Probability",
                f"{original_prob * 100:.2f}%"
            )

            st.write(class_name(original_pred))

        with col2:
            st.subheader("Modified Signal")

            st.metric(
                "Candidate Probability",
                f"{new_prob * 100:.2f}%",
                delta=(
                    f"{(new_prob-original_prob)*100:+.2f} "
                    "percentage points"
                )
            )

            st.write(class_name(new_pred))

        comparison = pd.DataFrame({
            "Scenario": ["Original", "Modified"],
            "Candidate Probability": [
                original_prob,
                new_prob
            ]
        })

        fig = px.bar(
            comparison,
            x="Scenario",
            y="Candidate Probability",
            color="Scenario",
            range_y=[0, 1],
            title="Prediction Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.warning(
            "This is a hypothetical ML sensitivity "
            "experiment. Modified values may not "
            "represent physically possible planets."
        )


# ----------------------------------------
# 9. Exoplanet Explorer
# ----------------------------------------

elif page == "Exoplanet Explorer":

    st.title("EXOPLANET EXPLORER")

    st.write(
        "Explore the Kepler dataset through "
        "interactive visualizations."
    )

    categories = st.multiselect(
        "Classification",
        ["Candidate / Confirmed", "False Positive"],
        default=[
            "Candidate / Confirmed",
            "False Positive"
        ]
    )

    filtered = df.copy()

    filtered["Category"] = filtered["target"].map({
        0: "False Positive",
        1: "Candidate / Confirmed"
    })

    filtered = filtered[
        filtered["Category"].isin(categories)
    ]

    st.metric(
        "Matching Objects",
        f"{len(filtered):,}"
    )

    fig = px.scatter(
        filtered,
        x="koi_period",
        y="koi_prad",
        color="Category",
        hover_data=["kepid", "koi_depth"],
        title="Orbital Period vs Estimated Planetary Radius",
        opacity=0.65
    )

    fig.update_xaxes(
        type="log",
        title="Orbital Period (days)"
    )

    fig.update_yaxes(
        type="log",
        title="Estimated Radius (Earth radii)"
    )

    fig.update_layout(
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        font_color="#e8f1ff"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Catalog Records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False)

    st.download_button(
        "Download Filtered Data",
        data=csv_data,
        file_name="planet_hunter_filtered.csv",
        mime="text/csv"
    )
