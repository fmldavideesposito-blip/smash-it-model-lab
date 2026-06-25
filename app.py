import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Config pagina
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smash IT Model Lab",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Smash IT Model Lab")
st.caption("Prediction Backtesting & Model Calibration")


# ------------------------------------------------------------
# Utility CSV
# ------------------------------------------------------------
def read_prediction_log(uploaded_file):
    """
    Legge il prediction_log.csv generato da Smash IT Optimizer.

    Il file viene esportato con:
    - separatore ;
    - decimale ,
    - encoding utf-8-sig
    """
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )

        # Se per qualche motivo viene caricata una sola colonna,
        # prova fallback automatico con separatore virgola.
        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                sep=",",
                decimal=".",
                encoding="utf-8-sig"
            )

        return df

    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def read_tennismylife_csv(uploaded_file):
    """
    Legge un CSV TennisMyLife.

    I file TennisMyLife annuali normalmente usano:
    - separatore ,
    - decimale .
    """
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=",",
            decimal=".",
            encoding="utf-8-sig"
        )

        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        return df

    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def show_dataframe_diagnostics(df: pd.DataFrame, title: str):
    """
    Mostra diagnostica semplice del dataframe caricato.
    """
    st.markdown(f"### {title}")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Rows", len(df))

    with c2:
        st.metric("Columns", len(df.columns))

    with st.expander("Columns found"):
        cols_df = pd.DataFrame(
            {
                "column_index": range(1, len(df.columns) + 1),
                "column_name": df.columns.tolist()
            }
        )

        st.dataframe(
            cols_df,
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------
# Tabs principali
# ------------------------------------------------------------
tab_pred, tab_summary, tab_actual, tab_backtest = st.tabs(
    [
        "Predictions",
        "Prediction Warehouse",
        "Actual Results",
        "Backtesting"
    ]
)


# ------------------------------------------------------------
# TAB 1 — Predictions
# ------------------------------------------------------------
with tab_pred:

    st.subheader("Prediction Log")

    prediction_file = st.file_uploader(
        "Upload prediction_log.csv",
        type=["csv"],
        key="pred"
    )

    if prediction_file:

        pred_df = read_prediction_log(prediction_file)

        st.session_state["prediction_log"] = pred_df

        st.success(
            f"{len(pred_df)} prediction rows loaded."
        )

        show_dataframe_diagnostics(
            pred_df,
            "Prediction Log Diagnostics"
        )

        st.markdown("### Preview")

        st.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True
        )

        required_cols = [
            "run_id",
            "run_timestamp",
            "model_version",
            "tournament",
            "year",
            "surface",
            "strategy",
            "player",
            "credits",
            "expected_points",
            "actual_points",
            "prediction_error",
            "efficiency_ratio",
        ]

        missing_cols = [
            c for c in required_cols
            if c not in pred_df.columns
        ]

        if missing_cols:
            st.warning(
                "Alcune colonne attese non sono presenti nel prediction_log.csv."
            )

            st.write(missing_cols)

        else:
            st.success(
                "Prediction log format looks valid."
            )

# ------------------------------------------------------------
# TAB 2 — Prediction Warehouse
# ------------------------------------------------------------
with tab_summary:

    st.subheader("Prediction Warehouse")

    st.caption(
        "Carica più prediction log scaricati dallo Smash IT Optimizer per creare uno storico centralizzato."
    )

    uploaded_logs = st.file_uploader(
        "Upload one or more prediction logs",
        type=["csv"],
        accept_multiple_files=True,
        key="prediction_warehouse"
    )

    if uploaded_logs:

        all_logs = []

        for f in uploaded_logs:

            try:

                df = read_prediction_log(f)

                df["source_file"] = f.name

                all_logs.append(df)

            except Exception:

                st.warning(
                    f"Unable to load {f.name}"
                )

        if all_logs:

            master_df = pd.concat(
                all_logs,
                ignore_index=True
            )

            st.session_state["prediction_log_master"] = master_df

            # ----------------------------------------------------
            # KPI
            # ----------------------------------------------------
            run_count = (
                master_df["run_id"].nunique()
                if "run_id" in master_df.columns
                else 0
            )

            tournament_count = (
                master_df["tournament"].nunique()
                if "tournament" in master_df.columns
                else 0
            )

            strategy_count = (
                master_df["strategy"].nunique()
                if "strategy" in master_df.columns
                else 0
            )

            rows_count = len(master_df)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Prediction Runs",
                    run_count
                )

            with c2:
                st.metric(
                    "Tournaments",
                    tournament_count
                )

            with c3:
                st.metric(
                    "Strategies",
                    strategy_count
                )

            with c4:
                st.metric(
                    "Rows",
                    rows_count
                )

            # ----------------------------------------------------
            # Tournament Summary
            # ----------------------------------------------------
            st.markdown("### Tournament Summary")

            summary_df = (
                master_df
                .groupby(
                    [
                        "run_id",
                        "tournament",
                        "year",
                        "surface",
                        "strategy",
                        "model_version"
                    ],
                    dropna=False
                )
                .agg(
                    players=("player", "count"),
                    total_expected_points=("expected_points", "sum"),
                    total_credits=("credits", "sum")
                )
                .reset_index()
            )

            summary_df = summary_df.sort_values(
                [
                    "year",
                    "tournament",
                    "strategy"
                ],
                ascending=[False, True, True]
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Most Selected Players
            # ----------------------------------------------------
            st.markdown("### Most Selected Players")

            player_summary = (
                master_df
                .groupby("player")
                .agg(
                    selections=("player", "count"),
                    avg_expected_points=("expected_points", "mean"),
                    avg_credits=("credits", "mean")
                )
                .reset_index()
            )

            player_summary = player_summary.sort_values(
                "selections",
                ascending=False
            )

            st.dataframe(
                player_summary.head(25),
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Strategy Snapshot
            # ----------------------------------------------------
            st.markdown("### Strategy Snapshot")

            strategy_summary = (
                master_df
                .groupby("strategy")
                .agg(
                    selections=("player", "count"),
                    avg_expected_points=("expected_points", "mean"),
                    total_expected_points=("expected_points", "sum")
                )
                .reset_index()
            )

            strategy_summary = strategy_summary.sort_values(
                "total_expected_points",
                ascending=False
            )

            st.dataframe(
                strategy_summary,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Full Warehouse
            # ----------------------------------------------------
            st.markdown("### Full Prediction Warehouse")

            st.dataframe(
                master_df,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Download
            # ----------------------------------------------------
            warehouse_csv = master_df.to_csv(
                index=False,
                sep=";",
                decimal=","
            ).encode("utf-8-sig")

            st.download_button(
                "⬇️ Download prediction_warehouse.csv",
                warehouse_csv,
                file_name="prediction_warehouse.csv",
                mime="text/csv",
                key="warehouse_download"
            )
# ------------------------------------------------------------
# TAB 2 — Actual Results
# ------------------------------------------------------------
with tab_actual:

    st.subheader("Actual Results")

    actual_file = st.file_uploader(
        "Upload TennisMyLife CSV",
        type=["csv"],
        key="actual"
    )

    if actual_file:

        actual_df = read_tennismylife_csv(actual_file)

        st.session_state["actual_results"] = actual_df

        st.success(
            f"{len(actual_df)} match rows loaded."
        )

        show_dataframe_diagnostics(
            actual_df,
            "TennisMyLife CSV Diagnostics"
        )

        st.markdown("### Preview")

        st.dataframe(
            actual_df.head(50),
            use_container_width=True,
            hide_index=True
        )

        expected_tml_cols = [
            "tourney_name",
            "tourney_date",
            "winner_name",
            "loser_name",
            "round",
            "score",
        ]

        available_tml_cols = [
            c for c in expected_tml_cols
            if c in actual_df.columns
        ]

        missing_tml_cols = [
            c for c in expected_tml_cols
            if c not in actual_df.columns
        ]

        st.markdown("### TennisMyLife Column Check")

        if available_tml_cols:
            st.success(
                "Relevant TennisMyLife columns found:"
            )
            st.write(available_tml_cols)

        if missing_tml_cols:
            st.warning(
                "Some expected TennisMyLife columns were not found:"
            )
            st.write(missing_tml_cols)


# ------------------------------------------------------------
# TAB 3 — Backtesting
# ------------------------------------------------------------
with tab_backtest:

    st.subheader("Backtesting")

    pred_ready = "prediction_log" in st.session_state
    actual_ready = "actual_results" in st.session_state

    c1, c2 = st.columns(2)

    with c1:
        if pred_ready:
            st.success("Prediction log loaded")
        else:
            st.info("Prediction log not loaded yet")

    with c2:
        if actual_ready:
            st.success("Actual results loaded")
        else:
            st.info("Actual results not loaded yet")

    if not pred_ready or not actual_ready:
        st.info(
            "Upload both prediction_log.csv and TennisMyLife CSV to enable backtesting."
        )

    else:
        pred_df = st.session_state["prediction_log"]
        actual_df = st.session_state["actual_results"]

        st.markdown("### Loaded Data Summary")

        s1, s2 = st.columns(2)

        with s1:
            st.metric(
                "Prediction rows",
                len(pred_df)
            )

        with s2:
            st.metric(
                "Actual match rows",
                len(actual_df)
            )

        st.info(
            "Prediction vs Actual comparison will be implemented in Model Lab V1.1."
        )

        st.markdown("### Next step preview")

        st.write(
            """
            The next release will:
            1. filter TennisMyLife matches by tournament and year;
            2. count wins for each predicted player;
            3. calculate actual points using `wins × 25`;
            4. compare expected points vs actual points;
            5. calculate prediction error and efficiency ratio.
            """
        )
