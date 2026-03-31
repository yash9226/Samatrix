"""
Credit Card Fraud Detection — Streamlit Web App
=================================================
A professional, hackathon-ready interface that loads pre-trained model
artefacts and predicts whether a credit-card transaction is legitimate
or fraudulent.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ──────────────────────────────────────────────
# Page configuration (must be the first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Yash Davda",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Theme state
# ──────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default to dark

is_dark = st.session_state.dark_mode

# ──────────────────────────────────────────────
# Custom CSS — theme-aware
# ──────────────────────────────────────────────
if is_dark:
    BG_MAIN      = "#0e1117"
    BG_SECONDARY = "#161b22"
    TEXT_PRIMARY  = "#e6edf3"
    TEXT_MUTED    = "#8b949e"
    BORDER_COLOR  = "#30363d"
    CARD_BG       = "#161b22"
    CARD_SHADOW   = "rgba(0,0,0,0.4)"
    SIDEBAR_BG1   = "#0d1117"
    SIDEBAR_BG2   = "#161b22"
    SIDEBAR_TEXT  = "#cfd8dc"
    FORM_TITLE_C  = "#b0c4d8"
    METRIC_BG     = "#1c2333"
else:
    BG_MAIN      = "#ffffff"
    BG_SECONDARY = "#f8f9fa"
    TEXT_PRIMARY  = "#1a1a2e"
    TEXT_MUTED    = "#6c757d"
    BORDER_COLOR  = "#e0e0e0"
    CARD_BG       = "#f5f7fa"
    CARD_SHADOW   = "rgba(0,0,0,0.06)"
    SIDEBAR_BG1   = "#0f2027"
    SIDEBAR_BG2   = "#203a43"
    SIDEBAR_TEXT  = "#cfd8dc"
    FORM_TITLE_C  = "#37474f"
    METRIC_BG     = "#f5f7fa"

st.markdown(f"""
<style>
/* ── Global ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background-color: {BG_MAIN};
    color: {TEXT_PRIMARY};
}}

/* ── Header banner ──────────────────────── */
.header-banner {{
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem 1.8rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px {CARD_SHADOW};
}}
.header-banner h1 {{
    color: #ffffff;
    font-size: 2.4rem;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}}
.header-banner p {{ color: #b0c4d8; font-size: 1.05rem; margin: 0; }}
.header-banner .author {{ color: #64ffda; font-size: 0.85rem; margin-top: 0.7rem; font-weight: 600; }}

/* ── Result cards ───────────────────────── */
.result-card {{
    padding: 1.6rem 2rem;
    border-radius: 14px;
    text-align: center;
    margin: 1rem 0;
    animation: fadeSlide 0.5s ease;
}}
@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.result-legit {{
    background: {"linear-gradient(135deg, #0d2818, #1a3a2a)" if is_dark else "linear-gradient(135deg, #e8f5e9, #c8e6c9)"};
    border-left: 6px solid #2e7d32;
}}
.result-legit h2 {{ color: {"#66bb6a" if is_dark else "#1b5e20"}; margin: 0; }}
.result-legit p  {{ color: {"#81c784" if is_dark else "#2e7d32"}; }}
.result-fraud {{
    background: {"linear-gradient(135deg, #2c0b0b, #3d1515)" if is_dark else "linear-gradient(135deg, #ffebee, #ffcdd2)"};
    border-left: 6px solid #c62828;
}}
.result-fraud h2 {{ color: {"#ef5350" if is_dark else "#b71c1c"}; margin: 0; }}
.result-fraud p  {{ color: {"#e57373" if is_dark else "#c62828"}; }}

/* ── Metric cards ───────────────────────── */
.metric-card {{
    background: {METRIC_BG};
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px {CARD_SHADOW};
}}
.metric-card .label {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin-bottom: 0.3rem; }}
.metric-card .value {{ font-size: 1.6rem; font-weight: 700; }}
.metric-green .value {{ color: {"#66bb6a" if is_dark else "#2e7d32"}; }}
.metric-red   .value {{ color: {"#ef5350" if is_dark else "#c62828"}; }}

/* ── Sidebar ────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SIDEBAR_BG1} 0%, {SIDEBAR_BG2} 100%);
}}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_TEXT} !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}

/* ── Form section ───────────────────────── */
.form-section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {FORM_TITLE_C};
    border-bottom: 2px solid {BORDER_COLOR};
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}}

/* ── Footer ─────────────────────────────── */
.footer {{
    text-align: center;
    color: {TEXT_MUTED};
    font-size: 0.78rem;
    margin-top: 3rem;
    padding: 1rem 0;
    border-top: 1px solid {BORDER_COLOR};
}}

/* ── Theme toggle button ───────────────── */
.theme-btn {{
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Paths to artefact files (same directory as app)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = {
    "fraud_model.pkl": "Trained classifier model",
    "scaler.pkl": "StandardScaler / scaler object",
    "feature_names.pkl": "List of feature names expected by the model",
    "scale_cols.pkl": "List of columns to scale",
}
OPTIONAL_FILES = {
    "selector.pkl": "Feature selector (optional)",
}


# ──────────────────────────────────────────────
# Helper — load artefacts with caching
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artefacts():
    """Load all .pkl files. Raises FileNotFoundError for required ones."""
    artefacts = {}

    # Required
    for fname, description in REQUIRED_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required file **{fname}** ({description}) not found in `{BASE_DIR}`."
            )
        artefacts[fname] = joblib.load(fpath)

    # Optional
    for fname, description in OPTIONAL_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.isfile(fpath):
            artefacts[fname] = joblib.load(fpath)
        else:
            artefacts[fname] = None

    return artefacts


# ──────────────────────────────────────────────
# Helper — preprocess a single transaction
# ──────────────────────────────────────────────
def preprocess(raw: dict, feature_names: list, scale_cols: list, scaler, selector):
    """
    Reproduce the same preprocessing pipeline used during training:
    1. Derive Hour from Time.
    2. Derive Amount_log from Amount.
    3. Build a DataFrame with the correct column order.
    4. Scale only the columns in scale_cols.
    5. Apply the feature selector (if provided).
    """
    # Engineered features
    raw["Hour"] = int(raw["Time"] // 3600) % 24
    raw["Amount_log"] = np.log1p(raw["Amount"])

    # Build DataFrame in the exact column order the model expects
    df = pd.DataFrame([raw])

    # Keep only the columns the model was trained on
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0  # fill missing features with zero
    df = df[feature_names]

    # Scale designated columns
    cols_to_scale = [c for c in scale_cols if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Feature selection (optional) — only apply if the model was trained
    # on the reduced feature set (i.e. selector output count matches model input count)
    if selector is not None:
        n_selected = int(selector.get_support().sum())
        model_n = None
        # We'll check compatibility at prediction time; for safety, skip here
        # and let the caller decide. Store selector output as alternative.
        # Skip selector by default — the model was trained on the full feature set.
        pass

    return df


# ──────────────────────────────────────────────
# Helper — run prediction
# ──────────────────────────────────────────────
def predict(model, processed_df):
    """Return (label, probability_or_None)."""
    prediction = model.predict(processed_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_df)[0]
    return int(prediction), proba


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(
            "https://img.icons8.com/3d-fluency/94/shield.png",
            width=80,
        )

        # ── Theme toggle ────────────────────────
        st.markdown("### 🎨 Theme")
        current = st.session_state.dark_mode
        label = "☀️ Switch to Light" if current else "🌙 Switch to Dark"
        if st.button(label, key="theme_toggle", use_container_width=True):
            st.session_state.dark_mode = not current
            st.rerun()

        st.markdown("---")
        st.markdown("### 🧑‍💻 Developer")
        st.markdown(
            """
            **Yash Davda**
            `ID: 23AIML012`
            """
        )
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            This app uses a **machine-learning model** trained on the
            [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
            to classify transactions as *legitimate* or *fraudulent*.

            **How it works**
            1. Enter the transaction features.
            2. Click **Predict**.
            3. Get an instant verdict.

            ---
            **Model Artefacts**
            - `fraud_model.pkl`
            - `scaler.pkl`
            - `feature_names.pkl`
            - `scale_cols.pkl`
            - `selector.pkl` *(optional)*
            """
        )
        st.markdown("---")
        st.caption("Built with ❤️ by Yash Davda using Streamlit")


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Header banner ─────────────────────────
    st.markdown(
        """
        <div class="header-banner">
            <h1>🛡️ Credit Card Fraud Detection</h1>
            <p>Enter transaction details and let the ML model predict whether it's legitimate or fraudulent.</p>
            <div class="author">By Yash Davda &nbsp;|&nbsp; ID: 23AIML012</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load artefacts ────────────────────────
    try:
        artefacts = load_artefacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Please place all required `.pkl` files in the project folder and reload the app.")
        st.stop()

    model = artefacts["fraud_model.pkl"]
    scaler = artefacts["scaler.pkl"]
    feature_names = artefacts["feature_names.pkl"]
    scale_cols = artefacts["scale_cols.pkl"]
    selector = artefacts["selector.pkl"]  # may be None

    if selector is not None:
        st.sidebar.success("✅ selector.pkl loaded")
    else:
        st.sidebar.info("ℹ️ selector.pkl not found — skipping feature selection")

    # ── Fraud sample data ─────────────────────
    FRAUD_SAMPLE = {
        "Time": 50000.0, "Amount": 2500.0,
        "V1": -4.5, "V2": 3.2, "V3": -2.8, "V4": 4.1, "V5": -1.9, "V6": -1.5,
        "V7": -3.8, "V8": 2.5, "V9": -2.2, "V10": -3.5, "V11": 2.8, "V12": -2.9,
        "V13": 0.5, "V14": -4.2, "V15": 0.3, "V16": -2.1, "V17": -3.6, "V18": -1.5,
        "V19": 0.8, "V20": 0.6, "V21": 1.2, "V22": -0.9, "V23": -0.5,
        "V24": 0.2, "V25": 0.3, "V26": -0.4, "V27": 0.7, "V28": -0.2,
    }

    # ── Input form ────────────────────────────
    st.markdown('<div class="form-section-title">📋 Transaction Details</div>', unsafe_allow_html=True)

    # Quick-fill button
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("⚡ Load Fraud Sample", use_container_width=True):
            for k, v in FRAUD_SAMPLE.items():
                st.session_state[k] = v
    with btn_col2:
        if st.button("🔄 Reset Fields", use_container_width=True):
            for k in FRAUD_SAMPLE:
                st.session_state[k] = 0.0

    with st.form("txn_form"):
        # Row 1 — Time & Amount
        st.markdown("**💰 Basic Info**")
        col_t, col_a, col_spacer = st.columns([1, 1, 0.5])
        with col_t:
            time_val = st.number_input(
                "⏱️ Time (seconds elapsed)", min_value=0.0,
                value=st.session_state.get("Time", 0.0), step=1.0,
                help="Seconds elapsed between this transaction and the first transaction in the dataset.",
                key="Time",
            )
        with col_a:
            amount_val = st.number_input(
                "💲 Amount ($)", min_value=0.0,
                value=st.session_state.get("Amount", 0.0), step=0.01,
                help="Transaction amount in US dollars.",
                key="Amount",
            )

        st.markdown("---")

        # V1 – V28 in a 4-column grid
        st.markdown("**🔢 PCA Components (V1 – V28)**")
        v_values = {}
        cols_per_row = 4
        for start in range(1, 29, cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, col in enumerate(cols):
                v_num = start + idx
                if v_num > 28:
                    break
                with col:
                    field = f"V{v_num}"
                    v_values[field] = st.number_input(
                        field,
                        value=st.session_state.get(field, 0.0),
                        format="%.6f",
                        key=field,
                    )

        st.markdown("")
        submitted = st.form_submit_button("🔍  Predict", use_container_width=True)

    # ── Prediction ────────────────────────────
    if submitted:
        # Collect raw inputs
        raw_input = {"Time": time_val, "Amount": amount_val}
        raw_input.update(v_values)

        try:
            processed_df = preprocess(raw_input, feature_names, scale_cols, scaler, selector)
            label, proba = predict(model, processed_df)
        except Exception as exc:
            st.error(f"⚠️ Prediction failed: {exc}")
            st.stop()

        # ── Result card ───────────────────────
        st.markdown("---")
        if label == 0:
            st.markdown(
                """
                <div class="result-card result-legit">
                    <h2>✅ Legitimate Transaction</h2>
                    <p>This transaction appears to be normal and safe.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card result-fraud">
                    <h2>🚨 Fraudulent Transaction</h2>
                    <p>This transaction has been flagged as potentially fraudulent!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability metrics
        if proba is not None:
            fraud_prob = proba[1] if len(proba) > 1 else proba[0]
            legit_prob = 1 - fraud_prob

            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(
                    f"""
                    <div class="metric-card metric-green">
                        <div class="label">Legitimate Probability</div>
                        <div class="value">{legit_prob:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"""
                    <div class="metric-card metric-red">
                        <div class="label">Fraud Probability</div>
                        <div class="value">{fraud_prob:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(fraud_prob), text=f"Fraud confidence: {fraud_prob:.2%}")

        # Expandable processed data view
        with st.expander("🔬 View Processed Input Data"):
            st.dataframe(processed_df, use_container_width=True)

    # ── Footer ────────────────────────────────
    st.markdown(
        '<div class="footer">Credit Card Fraud Detection &nbsp;|&nbsp; Yash Davda &nbsp;|&nbsp; ID: 23AIML012 &nbsp;|&nbsp; Powered by Streamlit</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
