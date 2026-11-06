
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Behavioral Economics Lab",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}
.hero {
    padding: 30px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background: rgba(128,128,128,.07);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL OUTPUTS
# ==========================================

@st.cache_data
def load_results():
    return pd.read_csv(OUTPUTS / "cv_results.csv")

@st.cache_data
def load_odds():
    return pd.read_csv(OUTPUTS / "odds_ratios.csv")

@st.cache_resource
def load_models():
    return joblib.load(OUTPUTS / "final_models.joblib")


results = load_results()
odds = load_odds()
models = load_models()

TARGETS = {
    "brand_switch": "Brand Switching",
    "snack_reaction": "Snack Price Reaction",
    "rec_follow": "Recommendation Following",
    "quality_top": "Quality as Top Factor"
}

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🧠 Behavioral Economics Lab")

st.sidebar.caption(
    "Consumer Behavior Research"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Research Findings",
        "Consumer Simulator",
        "Methodology"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Predictive analytics research platform")

# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.markdown("""
    <div class="hero">

    <h1>Behavioral Economics Lab</h1>

    <h3>Understanding the science of consumer decisions.</h3>

    An interactive research platform investigating
    how consumer preferences influence purchasing behavior.

    </div>
    """, unsafe_allow_html=True)

    st.subheader("Research Question")

    st.write(
        "How accurately can simple data models predict "
        "consumer purchasing behavior?"
    )

    logistic = results[
        results["model"] == "Logistic regression"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Outcomes Analyzed",
        len(TARGETS)
    )

    c2.metric(
        "Largest Outcome Sample",
        int(results["n"].max())
    )

    c3.metric(
        "Predictor Variables",
        7
    )

    st.subheader("Predictive Performance")

    display = logistic[
        ["target", "n", "roc_auc", "balanced_accuracy"]
    ].copy()

    display["target"] = display["target"].map(TARGETS)

    display.columns = [
        "Outcome",
        "Sample Size",
        "ROC-AUC",
        "Balanced Accuracy"
    ]

    st.dataframe(
        display.style.format({
            "ROC-AUC": "{:.3f}",
            "Balanced Accuracy": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These results describe predictions within the "
        "current survey sample. They should not be "
        "interpreted as proof of real-world purchasing behavior."
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("Model Performance")

    selected = st.selectbox(
        "Select an outcome",
        list(TARGETS.keys()),
        format_func=lambda x: TARGETS[x]
    )

    subset = results[
        results["target"] == selected
    ].copy()

    fig = px.bar(
        subset,
        x="model",
        y="roc_auc",
        color="model",
        text="roc_auc",
        title="ROC-AUC Model Comparison",
        labels={
            "model": "Model",
            "roc_auc": "ROC-AUC"
        }
    )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        annotation_text="Chance reference"
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detailed Metrics")

    st.dataframe(
        subset,
        use_container_width=True,
        hide_index=True
    )

    curve = OUTPUTS / "learning_curves.png"

    if curve.exists():

        st.subheader("Learning Curves")

        st.image(
            str(curve),
            use_container_width=True
        )

        st.caption(
            "Shaded regions represent fold-to-fold "
            "standard deviation, not 95% confidence intervals."
        )

# ==========================================
# RESEARCH FINDINGS
# ==========================================

elif page == "Research Findings":

    st.title("Research Findings")

    selected = st.selectbox(
        "Select an outcome",
        list(TARGETS.keys()),
        format_func=lambda x: TARGETS[x],
        key="odds_target"
    )

    subset = odds[
        odds["target"] == selected
    ].copy()

    st.subheader("Odds Ratios")

    st.dataframe(
        subset,
        use_container_width=True,
        hide_index=True
    )

    if not subset.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=subset["odds_ratio"],
                y=subset["feature"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=subset["ci_high"] - subset["odds_ratio"],
                    arrayminus=subset["odds_ratio"] - subset["ci_low"]
                )
            )
        )

        fig.add_vline(
            x=1,
            line_dash="dash",
            annotation_text="OR = 1"
        )

        fig.update_layout(
            title="Odds Ratios and Bootstrap Intervals",
            xaxis_title="Odds Ratio",
            yaxis_title="Predictor",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.warning(
        "Odds ratios represent associations, not causal effects. "
        "These exploratory estimates should be re-evaluated "
        "after the planned statistical improvements."
    )

# ==========================================
# CONSUMER SIMULATOR
# ==========================================

elif page == "Consumer Simulator":

    st.title("Consumer Behavior Simulator")

    st.write(
        "Create a hypothetical consumer profile and "
        "explore the predictions generated by the models."
    )

    with st.form("consumer_profile"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.selectbox(
                "Age Group",
                ["Under 18", "18-25", "26-40", "40+"]
            )

            frequency = st.selectbox(
                "Non-essential Purchase Frequency",
                [
                    "Rarely",
                    "Occasionally",
                    "Regularly",
                    "Frequently"
                ]
            )

            price = st.slider(
                "Price Importance",
                1, 5, 4
            )

            loyalty = st.slider(
                "Brand Loyalty",
                1, 5, 3
            )

        with col2:

            novelty = st.slider(
                "Willingness to Try New Products",
                1, 5, 3
            )

            cheaper = st.slider(
                "Preference for Cheaper Alternatives",
                1, 5, 4
            )

            recommendation = st.slider(
                "Likelihood of Following Recommendations",
                1, 5, 3
            )

        submitted = st.form_submit_button(
            "Generate Predictions",
            type="primary"
        )

    if submitted:

        frequency_mapping = {
            "Rarely": 1,
            "Occasionally": 2,
            "Regularly": 3,
            "Frequently": 4
        }

        profile = pd.DataFrame([{
            "age_18plus": int(age != "Under 18"),
            "freq_ord": frequency_mapping[frequency],
            "price": price,
            "loyal": loyalty,
            "novel": novelty,
            "cheaper": cheaper,
            "rectend": recommendation
        }])

        st.subheader("Predicted Outcomes")

        for target, artifact in models.items():

            model = artifact["model"]

            probability = model.predict_proba(
                profile
            )[0, 1]

            label = TARGETS.get(target, target)

            st.write(f"### {label}")

            st.progress(float(probability))

            st.write(
                f"Estimated probability: **{probability:.1%}**"
            )

            st.caption(
                f"Positive outcome: {artifact['positive_label']}"
            )

        st.info(
            "These predictions are based on the fitted survey models. "
            "They are exploratory estimates, not guarantees "
            "of individual purchasing behavior."
        )

# ==========================================
# METHODOLOGY
# ==========================================

elif page == "Methodology":

    st.title("Research Methodology")

    st.subheader("Research Question")

    st.write(
        "How accurately can simple data models predict "
        "consumer purchasing behavior?"
    )

    st.subheader("Predictor Variables")

    st.markdown("""
    - Age group
    - Non-essential purchase frequency
    - Price importance
    - Brand loyalty
    - Willingness to try unfamiliar products
    - Preference for cheaper alternatives
    - Likelihood of following recommendations
    """)

    st.subheader("Predictive Models")

    st.markdown("""
    - Logistic Regression
    - Random Forest
    - Class-Prior Baseline
    """)

    st.subheader("Evaluation")

    st.markdown("""
    - ROC-AUC
    - Accuracy
    - Balanced Accuracy
    - Brier Score
    - Repeated Stratified Cross-Validation
    - Permutation Testing
    - Bootstrap Intervals
    """)

    st.subheader("Limitations")

    st.markdown("""
    - Relatively small survey sample.
    - Sample disproportionately represents respondents under 18.
    - Hypothetical purchasing decisions may differ from actual purchases.
    - Predictive associations do not establish causation.
    - Model performance may vary across different populations.
    """)

    st.info(
        "The findings are preliminary and should be interpreted "
        "within the limitations of the survey design."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Behavioral Economics Lab | Independent Research Project"
)
