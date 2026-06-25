import io
import requests
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
# Costanti TennisMyLife
# ------------------------------------------------------------
TML_DATA_FILES_API = "https://stats.tennismylife.org/api/data-files"
POINTS_PER_WIN = 25


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


def read_tennismylife_bytes(content: bytes):
    """
    Legge un CSV TennisMyLife scaricato via URL.
    """
    buffer = io.BytesIO(content)

    try:
        df = pd.read_csv(
            buffer,
            sep=",",
            decimal=".",
            encoding="utf-8-sig"
        )

        if len(df.columns) == 1:
            buffer.seek(0)
            df = pd.read_csv(
                buffer,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        return df

    except Exception:
        buffer.seek(0)
        return pd.read_csv(buffer)


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


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    ).encode("utf-8-sig")


def ensure_numeric(df: pd.DataFrame, cols):
    """
    Converte in numerico le colonne indicate, se presenti.
    """
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce"
            )

    return out


# ------------------------------------------------------------
# Utility TennisMyLife Dynamic
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_tml_catalog():
    """
    Scarica il catalogo dinamico dei file TennisMyLife.
    """
    response = requests.get(
        TML_DATA_FILES_API,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    files = data.get("files", [])

    return pd.DataFrame(files)


@st.cache_data(show_spinner=False)
def download_tml_csv_from_url(url: str):
    """
    Scarica un CSV TennisMyLife da URL e lo converte in dataframe.
    """
    response = requests.get(
        url,
        timeout=60
    )
    response.raise_for_status()

    return read_tennismylife_bytes(response.content)


def find_tml_season_url(catalog_df: pd.DataFrame, year: int):
    """
    Trova nel catalogo TennisMyLife il file ATP annuale, es. 2026.csv.

    Evita i file Challenger quando possibile.
    """
    if catalog_df.empty:
        return None, None

    if "name" not in catalog_df.columns or "url" not in catalog_df.columns:
        return None, None

    target_name = f"{int(year)}.csv"

    exact = catalog_df[
        catalog_df["name"].astype(str).str.lower() == target_name.lower()
    ].copy()

    if not exact.empty:
        row = exact.iloc[0]
        return row["url"], row["name"]

    contains = catalog_df[
        catalog_df["name"].astype(str).str.contains(str(year), case=False, na=False)
        & ~catalog_df["name"].astype(str).str.contains("challenger", case=False, na=False)
        & catalog_df["name"].astype(str).str.endswith(".csv")
    ].copy()

    if not contains.empty:
        row = contains.iloc[0]
        return row["url"], row["name"]

    return None, None


def get_years_from_prediction_data():
    """
    Estrae gli anni disponibili dal Prediction Warehouse o dal singolo prediction_log.
    """
    if "prediction_log_master" in st.session_state:
        df = st.session_state["prediction_log_master"]
    elif "prediction_log" in st.session_state:
        df = st.session_state["prediction_log"]
    else:
        return []

    if "year" not in df.columns:
        return []

    years = (
        pd.to_numeric(df["year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return sorted(years)


# ------------------------------------------------------------
# Utility Actual Results
# ------------------------------------------------------------
def build_actual_tournament_summary(actual_df: pd.DataFrame):
    """
    Costruisce un riepilogo per torneo dai dati TennisMyLife.
    """
    if "tourney_name" not in actual_df.columns:
        return pd.DataFrame()

    agg_dict = {}

    if "match_num" in actual_df.columns:
        agg_dict["matches"] = ("match_num", "count")
    else:
        agg_dict["matches"] = ("tourney_name", "count")

    if "winner_name" in actual_df.columns:
        agg_dict["unique_winners"] = ("winner_name", "nunique")

    if "loser_name" in actual_df.columns:
        agg_dict["unique_losers"] = ("loser_name", "nunique")

    if "surface" in actual_df.columns:
        agg_dict["surface"] = ("surface", "first")

    if "draw_size" in actual_df.columns:
        agg_dict["draw_size"] = ("draw_size", "first")

    if "tourney_level" in actual_df.columns:
        agg_dict["tourney_level"] = ("tourney_level", "first")

    if "tourney_date" in actual_df.columns:
        agg_dict["first_date"] = ("tourney_date", "min")
        agg_dict["last_date"] = ("tourney_date", "max")

    summary = (
        actual_df
        .groupby("tourney_name", dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("matches", ascending=False)
    )

    return summary


def build_actual_player_wins(actual_df: pd.DataFrame):
    """
    Calcola wins e actual points per player.
    """
    if "winner_name" not in actual_df.columns:
        return pd.DataFrame()

    base = actual_df.copy()

    group_cols = ["winner_name"]

    agg_dict = {
        "wins": ("winner_name", "count")
    }

    if "tourney_name" in base.columns:
        agg_dict["tournaments_won_matches"] = ("tourney_name", "nunique")

    if "surface" in base.columns:
        agg_dict["surfaces"] = ("surface", lambda x: ", ".join(sorted(set(x.dropna().astype(str)))))

    if "round" in base.columns:
        agg_dict["rounds_won"] = ("round", lambda x: ", ".join(sorted(set(x.dropna().astype(str)))))

    player_wins = (
        base
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"winner_name": "player"})
    )

    player_wins["actual_points"] = player_wins["wins"] * POINTS_PER_WIN

    player_wins = player_wins.sort_values(
        ["actual_points", "wins"],
        ascending=[False, False]
    )

    return player_wins


def filter_actuals_by_tournament(actual_df: pd.DataFrame, tournament_filter: str):
    """
    Filtra actual_df per torneo se richiesto.
    """
    if tournament_filter == "All Tournaments":
        return actual_df.copy()

    if "tourney_name" not in actual_df.columns:
        return actual_df.copy()

    return actual_df[
        actual_df["tourney_name"].astype(str) == tournament_filter
    ].copy()


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

        pred_df = ensure_numeric(
            pred_df,
            [
                "year",
                "budget",
                "team_size",
                "credits",
                "expected_points",
                "value_index",
                "model_rank",
                "players_found",
                "matched",
                "match_rate",
                "q_count",
                "wc_count",
                "ll_count",
            ]
        )

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

                df = ensure_numeric(
                    df,
                    [
                        "year",
                        "budget",
                        "team_size",
                        "credits",
                        "expected_points",
                        "value_index",
                        "model_rank",
                        "players_found",
                        "matched",
                        "match_rate",
                        "q_count",
                        "wc_count",
                        "ll_count",
                    ]
                )

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
                st.metric("Prediction Runs", run_count)

            with c2:
                st.metric("Tournaments", tournament_count)

            with c3:
                st.metric("Strategies", strategy_count)

            with c4:
                st.metric("Rows", rows_count)

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
            # Most Selected Players Overall
            # ----------------------------------------------------
            st.markdown("### Most Selected Players Overall")

            strategy_options = (
                ["All Strategies"]
                + sorted(master_df["strategy"].dropna().unique().tolist())
                if "strategy" in master_df.columns
                else ["All Strategies"]
            )

            selected_strategy_filter = st.selectbox(
                "Filter by strategy",
                strategy_options,
                key="most_selected_strategy_filter"
            )

            if selected_strategy_filter != "All Strategies":
                player_base_df = master_df[
                    master_df["strategy"] == selected_strategy_filter
                ].copy()
            else:
                player_base_df = master_df.copy()

            if not player_base_df.empty:

                player_summary = (
                    player_base_df
                    .groupby("player", dropna=False)
                    .agg(
                        selections=("player", "count"),
                        tournaments_selected=("tournament", "nunique"),
                        strategies_selected=("strategy", "nunique"),
                        avg_expected_points=("expected_points", "mean"),
                        total_expected_points=("expected_points", "sum"),
                        max_expected_points=("expected_points", "max"),
                        avg_credits=("credits", "mean"),
                        min_credits=("credits", "min"),
                        max_credits=("credits", "max"),
                    )
                    .reset_index()
                )

                player_summary["selection_share_pct"] = (
                    player_summary["selections"]
                    / len(player_base_df)
                    * 100
                )

                player_summary = player_summary.sort_values(
                    [
                        "selections",
                        "total_expected_points"
                    ],
                    ascending=[False, False]
                )

                for col in [
                    "avg_expected_points",
                    "total_expected_points",
                    "max_expected_points",
                    "avg_credits",
                    "min_credits",
                    "max_credits",
                    "selection_share_pct",
                ]:
                    if col in player_summary.columns:
                        player_summary[col] = player_summary[col].round(2)

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Unique Players",
                        player_summary["player"].nunique()
                    )

                with c2:
                    top_player = (
                        player_summary.iloc[0]["player"]
                        if len(player_summary) > 0
                        else "-"
                    )

                    st.metric(
                        "Most Selected",
                        top_player
                    )

                with c3:
                    top_selections = (
                        int(player_summary.iloc[0]["selections"])
                        if len(player_summary) > 0
                        else 0
                    )

                    st.metric(
                        "Top Selections",
                        top_selections
                    )

                st.dataframe(
                    player_summary,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download most_selected_players.csv",
                    dataframe_to_csv_bytes(player_summary),
                    file_name="most_selected_players.csv",
                    mime="text/csv",
                    key="download_most_selected_players"
                )

            else:
                st.info("No player data available for the selected strategy.")

            # ----------------------------------------------------
            # Strategy Snapshot
            # ----------------------------------------------------
            st.markdown("### Strategy Snapshot")

            strategy_summary = (
                master_df
                .groupby("strategy", dropna=False)
                .agg(
                    selections=("player", "count"),
                    avg_expected_points=("expected_points", "mean"),
                    total_expected_points=("expected_points", "sum"),
                    avg_credits=("credits", "mean"),
                    total_credits=("credits", "sum"),
                )
                .reset_index()
            )

            strategy_summary = strategy_summary.sort_values(
                "total_expected_points",
                ascending=False
            )

            for col in [
                "avg_expected_points",
                "total_expected_points",
                "avg_credits",
                "total_credits",
            ]:
                if col in strategy_summary.columns:
                    strategy_summary[col] = strategy_summary[col].round(2)

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
            # Download Warehouse
            # ----------------------------------------------------
            st.download_button(
                "⬇️ Download prediction_warehouse.csv",
                dataframe_to_csv_bytes(master_df),
                file_name="prediction_warehouse.csv",
                mime="text/csv",
                key="warehouse_download"
            )


# ------------------------------------------------------------
# TAB 3 — Actual Results
# ------------------------------------------------------------
with tab_actual:

    st.subheader("Actual Results")

    source_mode = st.radio(
        "Source",
        [
            "TennisMyLife Dynamic",
            "Manual CSV Upload"
        ],
        horizontal=True,
        key="actual_source_mode"
    )

    if source_mode == "TennisMyLife Dynamic":

        st.caption(
            "Scarica automaticamente i dati ATP annuali da TennisMyLife usando il catalogo dinamico."
        )

        warehouse_years = get_years_from_prediction_data()

        if warehouse_years:
            st.success(
                f"Anni rilevati dal Prediction Warehouse: {warehouse_years}"
            )
            default_years = warehouse_years
        else:
            st.info(
                "Nessun anno rilevato dal Prediction Warehouse. Seleziona manualmente la stagione."
            )
            default_years = [2026]

        selectable_years = list(range(2026, 2019, -1))

        selected_years = st.multiselect(
            "Season years to load",
            selectable_years,
            default=[
                y for y in default_years
                if y in selectable_years
            ],
            key="tml_dynamic_years"
        )

        show_catalog = st.checkbox(
            "Show TennisMyLife catalog",
            value=False,
            key="show_tml_catalog"
        )

        if st.button(
            "Load Actual Results from TennisMyLife",
            key="load_tml_dynamic"
        ):

            if not selected_years:
                st.warning("Seleziona almeno un anno.")
            else:
                try:
                    with st.spinner("Loading TennisMyLife catalog..."):
                        catalog_df = fetch_tml_catalog()

                    if show_catalog:
                        st.markdown("### TennisMyLife Catalog")
                        st.dataframe(
                            catalog_df,
                            use_container_width=True,
                            hide_index=True
                        )

                    loaded_actuals = []
                    load_report = []

                    for y in selected_years:
                        url, file_name = find_tml_season_url(
                            catalog_df,
                            y
                        )

                        if not url:
                            load_report.append(
                                {
                                    "year": y,
                                    "file": "",
                                    "status": "not found",
                                    "rows": 0,
                                }
                            )
                            continue

                        with st.spinner(f"Loading TennisMyLife {file_name}..."):
                            year_df = download_tml_csv_from_url(url)

                        year_df["source_year"] = y
                        year_df["source_file"] = file_name
                        year_df["source_url"] = url

                        loaded_actuals.append(year_df)

                        load_report.append(
                            {
                                "year": y,
                                "file": file_name,
                                "status": "loaded",
                                "rows": len(year_df),
                            }
                        )

                    if loaded_actuals:
                        actual_df = pd.concat(
                            loaded_actuals,
                            ignore_index=True
                        )

                        st.session_state["actual_results"] = actual_df

                        st.success(
                            f"{len(actual_df)} actual match rows loaded from TennisMyLife."
                        )

                        st.markdown("### Load Report")

                        st.dataframe(
                            pd.DataFrame(load_report),
                            use_container_width=True,
                            hide_index=True
                        )

                        show_dataframe_diagnostics(
                            actual_df,
                            "TennisMyLife Dynamic Data Diagnostics"
                        )

                        st.markdown("### Preview")

                        st.dataframe(
                            actual_df.head(50),
                            use_container_width=True,
                            hide_index=True
                        )

                    else:
                        st.error(
                            "Nessun file TennisMyLife caricato."
                        )

                        st.dataframe(
                            pd.DataFrame(load_report),
                            use_container_width=True,
                            hide_index=True
                        )

                except Exception as e:
                    st.error(
                        "Errore durante il caricamento dinamico dei dati TennisMyLife."
                    )
                    st.exception(e)

    else:

        st.caption(
            "Modalità fallback: carica manualmente un CSV TennisMyLife."
        )

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

    # --------------------------------------------------------
    # Actual Results Analysis
    # --------------------------------------------------------
    if "actual_results" in st.session_state:

        actual_df = st.session_state["actual_results"]

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

        # ----------------------------------------------------
        # Filters
        # ----------------------------------------------------
        st.markdown("### Actual Results Filters")

        if "tourney_name" in actual_df.columns:
            tournament_options = (
                ["All Tournaments"]
                + sorted(
                    actual_df["tourney_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
        else:
            tournament_options = ["All Tournaments"]

        selected_actual_tournament = st.selectbox(
            "Filter by tournament",
            tournament_options,
            key="actual_tournament_filter"
        )

        filtered_actual_df = filter_actuals_by_tournament(
            actual_df,
            selected_actual_tournament
        )

        player_search = st.text_input(
            "Search player in actual winners",
            value="",
            key="actual_player_search"
        )

        # ----------------------------------------------------
        # Tournament Summary
        # ----------------------------------------------------
        tournament_summary = build_actual_tournament_summary(
            filtered_actual_df
        )

        if not tournament_summary.empty:
            st.markdown("### Actual Tournament Summary")

            st.dataframe(
                tournament_summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download actual_tournament_summary.csv",
                dataframe_to_csv_bytes(tournament_summary),
                file_name="actual_tournament_summary.csv",
                mime="text/csv",
                key="download_actual_tournament_summary"
            )

        # ----------------------------------------------------
        # Actual Player Wins
        # ----------------------------------------------------
        player_wins = build_actual_player_wins(
            filtered_actual_df
        )

        if not player_wins.empty:

            if player_search.strip():
                player_wins = player_wins[
                    player_wins["player"]
                    .astype(str)
                    .str.contains(
                        player_search.strip(),
                        case=False,
                        na=False
                    )
                ].copy()

            st.markdown("### Actual Player Wins")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Players with wins",
                    player_wins["player"].nunique()
                )

            with c2:
                total_wins = int(player_wins["wins"].sum()) if "wins" in player_wins.columns else 0

                st.metric(
                    "Total wins",
                    total_wins
                )

            with c3:
                total_actual_points = int(player_wins["actual_points"].sum()) if "actual_points" in player_wins.columns else 0

                st.metric(
                    "Total actual points",
                    total_actual_points
                )

            st.dataframe(
                player_wins,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download actual_player_wins.csv",
                dataframe_to_csv_bytes(player_wins),
                file_name="actual_player_wins.csv",
                mime="text/csv",
                key="download_actual_player_wins"
            )


# ------------------------------------------------------------
# TAB 4 — Backtesting
# ------------------------------------------------------------
with tab_backtest:

    st.subheader("Backtesting")

    pred_ready = (
        "prediction_log" in st.session_state
        or "prediction_log_master" in st.session_state
    )

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
            "Upload prediction logs and load TennisMyLife actual results to enable backtesting."
        )

    else:
        if "prediction_log_master" in st.session_state:
            pred_df = st.session_state["prediction_log_master"]
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
