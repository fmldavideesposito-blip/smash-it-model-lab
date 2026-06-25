import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smash IT Model Lab",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Smash IT Model Lab")
st.caption("Prediction Backtesting & Model Calibration")

tab_pred, tab_actual, tab_backtest = st.tabs(
    [
        "Predictions",
        "Actual Results",
        "Backtesting"
    ]
)

with tab_pred:

    st.subheader("Prediction Log")

    prediction_file = st.file_uploader(
        "Upload prediction_log.csv",
        type=["csv"],
        key="pred"
    )

    if prediction_file:

        pred_df = pd.read_csv(prediction_file)

        st.success(
            f"{len(pred_df)} prediction rows loaded."
        )

        st.dataframe(
            pred_df,
            use_container_width=True
        )

with tab_actual:

    st.subheader("Actual Results")

    actual_file = st.file_uploader(
        "Upload TennisMyLife CSV",
        type=["csv"],
        key="actual"
    )

    if actual_file:

        actual_df = pd.read_csv(actual_file)

        st.success(
            f"{len(actual_df)} match rows loaded."
        )

        st.dataframe(
            actual_df.head(20),
            use_container_width=True
        )

with tab_backtest:

    st.subheader("Backtesting")

    st.info(
        "Prediction vs Actual comparison will be implemented in V1.1"
    )
