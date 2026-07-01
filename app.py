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

# ------------------------------------------------------------
# Prediction vs Actual Global
# ------------------------------------------------------------
def build_prediction_vs_actual_global(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame
):

    if "player" not in pred_df.columns:
        return pd.DataFrame()

    if "expected_points" not in pred_df.columns:
        return pd.DataFrame()

    actual_wins = build_actual_player_wins(actual_df)

    if actual_wins.empty:
        return pd.DataFrame()

    prediction_summary = (
        pred_df
        .groupby(
            ["player"],
            dropna=False
        )
        .agg(
            expected_points=("expected_points", "sum"),
            selections=("player", "count")
        )
        .reset_index()
    )

    prediction_summary["player_norm"] = (
        prediction_summary["player"]
        .apply(normalize_player_name)
        )

    actual_wins["player_norm"] = (
        actual_wins["player"]
        .apply(normalize_player_name)
    )

    merged = pd.merge(
        prediction_summary,
        actual_wins[
            [
                "player_norm",
                "wins",
                "actual_points"
            ]
        ],
        on="player_norm",
        how="left"
    )

    merged["wins"] = merged["wins"].fillna(0)
    merged["actual_points"] = merged["actual_points"].fillna(0)

    merged["prediction_error"] = (
        merged["actual_points"]
        - merged["expected_points"]
    )

    merged["efficiency_ratio"] = (
        merged["actual_points"]
        / merged["expected_points"]
    )

    merged["efficiency_ratio"] = (
        merged["efficiency_ratio"]
        .replace([float("inf")], 0)
        .fillna(0)
    )

    merged = merged.sort_values(
        "prediction_error",
        ascending=False
    )

    merged["expected_points"] = (
        merged["expected_points"]
        .round(1)
    )

    merged["prediction_error"] = (
        merged["prediction_error"]
        .round(1)
    )

    merged["efficiency_ratio"] = (
        merged["efficiency_ratio"]
        .round(2)
    )

    merged["actual_minus_expected"] = (
        merged["actual_points"]
        -
        merged["expected_points"]
    )

    return merged

def filter_actuals_by_tournament(
    actual_df: pd.DataFrame,
    tournament_filter: str
):
    """
    Filtra actual_df per torneo se richiesto.
    """

    if tournament_filter == "All Tournaments":
        return actual_df.copy()

    if "tourney_name" not in actual_df.columns:
        return actual_df.copy()

    return actual_df[
        actual_df["tourney_name"].astype(str)
        == tournament_filter
    ].copy()


# ------------------------------------------------------------
# Tournament Mapping
# ------------------------------------------------------------
def normalize_tournament_name(name):
    """
    Normalizza il nome torneo per confronti robusti.
    """

    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    replacements = {
        "roland garros": "rolandgarros",
        "french open": "rolandgarros",
        "rome": "roma",
        "rome masters": "roma",
        "internazionali bnl d'italia": "roma",
        "madrid masters": "madrid",
    }

    return replacements.get(
        name,
        name.replace(" ", "")
    )


def build_tournament_mapping(pred_df, actual_df):
    """
    Costruisce una tabella di mapping tra i nomi torneo del Prediction Warehouse
    e i nomi torneo presenti nel database TennisMyLife.
    """

    pred_tournaments = []

    if "tournament" in pred_df.columns:
        pred_tournaments = sorted(
            pred_df["tournament"]
            .dropna()
            .astype(str)
            .unique()
        )

    actual_tournaments = []

    if "tourney_name" in actual_df.columns:
        actual_tournaments = sorted(
            actual_df["tourney_name"]
            .dropna()
            .astype(str)
            .unique()
        )

    mapping_rows = []

    for pred_name in pred_tournaments:

        pred_norm = normalize_tournament_name(pred_name)

        matches = []

        for actual_name in actual_tournaments:

            actual_norm = normalize_tournament_name(actual_name)

            if pred_norm == actual_norm:
                matches.append(actual_name)

        mapping_rows.append(
            {
                "prediction_tournament": pred_name,
                "matched_actual_tournaments": ", ".join(matches),
                "matched_count": len(matches),
            }
        )

    return pd.DataFrame(mapping_rows)


# ------------------------------------------------------------
# Player Matching between Predictions and Actuals
# ------------------------------------------------------------
def normalize_player_name(name):
    """
    Normalizza il nome player per confronti robusti.
    """

    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    replacements = {
        "-": " ",
        ".": "",
        "'": "",
        "’": "",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = " ".join(name.split())

    return name


def build_predicted_players_actual_match(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame
):
    """
    Cerca tutti i player del Prediction Warehouse negli actual TennisMyLife.
    """

    if "player" not in pred_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    if "winner_name" not in actual_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    pred_players = (
        pred_df["player"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    actual_wins = build_actual_player_wins(actual_df)

    if actual_wins.empty:
        return pd.DataFrame(), pd.DataFrame()

    actual_wins = actual_wins.copy()

    actual_wins["player_norm"] = actual_wins["player"].apply(
        normalize_player_name
    )

    rows = []

    for player in pred_players:

        player_norm = normalize_player_name(player)

        matched = actual_wins[
            actual_wins["player_norm"] == player_norm
        ].copy()

        if not matched.empty:

            row = matched.iloc[0].to_dict()

            rows.append(
                {
                    "predicted_player": player,
                    "matched_actual_player": row.get("player", ""),
                    "found_in_actuals": True,
                    "wins": row.get("wins", 0),
                    "actual_points": row.get("actual_points", 0),
                    "tournaments_won_matches": row.get(
                        "tournaments_won_matches",
                        ""
                    ),
                    "surfaces": row.get("surfaces", ""),
                    "rounds_won": row.get("rounds_won", ""),
                }
            )

        else:

            rows.append(
                {
                    "predicted_player": player,
                    "matched_actual_player": "",
                    "found_in_actuals": False,
                    "wins": 0,
                    "actual_points": 0,
                    "tournaments_won_matches": "",
                    "surfaces": "",
                    "rounds_won": "",
                }
            )

    match_df = pd.DataFrame(rows)

    unmatched_df = match_df[
        match_df["found_in_actuals"] == False
    ].copy()

    return match_df, unmatched_df

# ------------------------------------------------------------
# Prediction vs Actual by Tournament
# ------------------------------------------------------------
def build_prediction_vs_actual_tournament(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame
):
    """
    Confronta le previsioni con gli actual TennisMyLife a livello torneo.

    Logica:
    - prende ogni riga del prediction warehouse;
    - usa tournament/year/player/strategy;
    - mappa il nome torneo prediction con il nome torneo TennisMyLife;
    - conta le vittorie reali solo in quel torneo;
    - calcola actual_points = wins * POINTS_PER_WIN;
    - calcola prediction_error = actual_points - expected_points;
    - calcola efficiency_ratio = actual_points / expected_points.
    """

    required_pred_cols = [
        "tournament",
        "year",
        "strategy",
        "player",
        "expected_points"
    ]

    for col in required_pred_cols:
        if col not in pred_df.columns:
            return pd.DataFrame(), pd.DataFrame()

    if "tourney_name" not in actual_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    if "winner_name" not in actual_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    pred_base = pred_df.copy()
    actual_base = actual_df.copy()

    pred_base["expected_points"] = pd.to_numeric(
        pred_base["expected_points"],
        errors="coerce"
    ).fillna(0)

    pred_base["year"] = pd.to_numeric(
        pred_base["year"],
        errors="coerce"
    ).fillna(0).astype(int)

    # --------------------------------------------------------
    # Normalizzazioni
    # --------------------------------------------------------
    pred_base["prediction_tournament_norm"] = (
        pred_base["tournament"]
        .apply(normalize_tournament_name)
    )

    pred_base["player_norm"] = (
        pred_base["player"]
        .apply(normalize_player_name)
    )

    actual_base["actual_tournament_norm"] = (
        actual_base["tourney_name"]
        .apply(normalize_tournament_name)
    )

    actual_base["winner_norm"] = (
        actual_base["winner_name"]
        .apply(normalize_player_name)
    )

    if "loser_name" in actual_base.columns:
        actual_base["loser_norm"] = (
            actual_base["loser_name"]
            .apply(normalize_player_name)
        )
    else:
        actual_base["loser_norm"] = ""

    # --------------------------------------------------------
    # Anno actual
    # --------------------------------------------------------
    if "source_year" in actual_base.columns:
        actual_base["actual_year"] = pd.to_numeric(
            actual_base["source_year"],
            errors="coerce"
        ).fillna(0).astype(int)

    elif "tourney_date" in actual_base.columns:
        actual_base["actual_year"] = (
            actual_base["tourney_date"]
            .astype(str)
            .str.slice(0, 4)
        )

        actual_base["actual_year"] = pd.to_numeric(
            actual_base["actual_year"],
            errors="coerce"
        ).fillna(0).astype(int)

    else:
        actual_base["actual_year"] = 0

    # --------------------------------------------------------
    # Aggregate prediction by tournament / strategy / player
    # --------------------------------------------------------
    group_cols = [
        "run_id",
        "tournament",
        "year",
        "surface",
        "strategy",
        "player",
        "prediction_tournament_norm",
        "player_norm"
    ]

    existing_group_cols = [
        c for c in group_cols
        if c in pred_base.columns
    ]

    prediction_summary = (
        pred_base
        .groupby(
            existing_group_cols,
            dropna=False
        )
        .agg(
            expected_points=("expected_points", "sum"),
            selections=("player", "count")
        )
        .reset_index()
    )

    rows = []

    for _, pred_row in prediction_summary.iterrows():

        pred_tournament = pred_row.get("tournament", "")
        pred_tournament_norm = pred_row.get(
            "prediction_tournament_norm",
            ""
        )

        pred_year = int(pred_row.get("year", 0))
        pred_player = pred_row.get("player", "")
        pred_player_norm = pred_row.get("player_norm", "")
        pred_strategy = pred_row.get("strategy", "")
        expected_points = float(pred_row.get("expected_points", 0))

        # ----------------------------------------------------
        # Filtra actual per torneo + anno
        # ----------------------------------------------------
        actual_tournament_df = actual_base[
            actual_base["actual_tournament_norm"]
            == pred_tournament_norm
        ].copy()

        if pred_year > 0 and "actual_year" in actual_tournament_df.columns:
            actual_tournament_df = actual_tournament_df[
                actual_tournament_df["actual_year"] == pred_year
            ].copy()

        # ----------------------------------------------------
        # Calcola wins nel torneo
        # ----------------------------------------------------
        wins = int(
            (
                actual_tournament_df["winner_norm"]
                == pred_player_norm
            ).sum()
        )

        if "loser_norm" in actual_tournament_df.columns:
            losses = int(
                (
                    actual_tournament_df["loser_norm"]
                    == pred_player_norm
                ).sum()
            )
        else:
            losses = 0

        matches_played = wins + losses
        actual_points = wins * POINTS_PER_WIN

        prediction_error = actual_points - expected_points

        efficiency_ratio = (
            actual_points / expected_points
            if expected_points > 0
            else 0
        )

        rounds_won = ""

        if wins > 0 and "round" in actual_tournament_df.columns:
            rounds_won = ", ".join(
                sorted(
                    actual_tournament_df.loc[
                        actual_tournament_df["winner_norm"]
                        == pred_player_norm,
                        "round"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

        rows.append(
            {
                "run_id": pred_row.get("run_id", ""),
                "tournament": pred_tournament,
                "year": pred_year,
                "surface": pred_row.get("surface", ""),
                "strategy": pred_strategy,
                "player": pred_player,
                "expected_points": round(expected_points, 2),
                "actual_wins": wins,
                "actual_losses": losses,
                "actual_matches_played": matches_played,
                "actual_points": round(actual_points, 2),
                "prediction_error": round(prediction_error, 2),
                "efficiency_ratio": round(efficiency_ratio, 3),
                "rounds_won": rounds_won,
                "matched_actual_tournament": (
                    actual_tournament_df["tourney_name"].iloc[0]
                    if not actual_tournament_df.empty
                    else ""
                ),
                "actual_matches_in_tournament": len(actual_tournament_df),
            }
        )

    detail_df = pd.DataFrame(rows)

    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --------------------------------------------------------
    # Team / Strategy summary
    # --------------------------------------------------------
    summary_df = (
        detail_df
        .groupby(
            [
                "run_id",
                "tournament",
                "year",
                "surface",
                "strategy"
            ],
            dropna=False
        )
        .agg(
            players=("player", "count"),
            expected_points=("expected_points", "sum"),
            actual_points=("actual_points", "sum"),
            actual_wins=("actual_wins", "sum"),
            prediction_error=("prediction_error", "sum")
        )
        .reset_index()
    )

    summary_df["efficiency_ratio"] = (
        summary_df["actual_points"]
        / summary_df["expected_points"]
    )

    summary_df["efficiency_ratio"] = (
        summary_df["efficiency_ratio"]
        .replace([float("inf")], 0)
        .fillna(0)
    )

    for col in [
        "expected_points",
        "actual_points",
        "prediction_error",
        "efficiency_ratio"
    ]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)

    detail_df = detail_df.sort_values(
        [
            "tournament",
            "strategy",
            "prediction_error"
        ],
        ascending=[True, True, False]
    )

    summary_df = summary_df.sort_values(
        [
            "tournament",
            "efficiency_ratio"
        ],
        ascending=[True, False]
    )

    return detail_df, summary_df

def enrich_prediction_warehouse_with_actuals(
    pred_df: pd.DataFrame,
    tournament_detail_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Arricchisce il Prediction Warehouse con gli actual calcolati
    da build_prediction_vs_actual_tournament().

    Riempie:
    - actual_points
    - actual_wins
    - actual_matches_played
    - actual_best_round
    - prediction_error
    - efficiency_ratio
    """

    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    if tournament_detail_df is None or tournament_detail_df.empty:
        return pred_df.copy()

    warehouse = pred_df.copy()
    detail = tournament_detail_df.copy()

    # --------------------------------------------------------
    # Normalizzazione colonne chiave
    # --------------------------------------------------------
    if "year" in warehouse.columns:
        warehouse["year"] = pd.to_numeric(
            warehouse["year"],
            errors="coerce"
        ).fillna(0).astype(int)

    if "year" in detail.columns:
        detail["year"] = pd.to_numeric(
            detail["year"],
            errors="coerce"
        ).fillna(0).astype(int)

    # --------------------------------------------------------
    # Chiavi di merge
    # --------------------------------------------------------
    preferred_keys = [
        "run_id",
        "tournament",
        "year",
        "surface",
        "strategy",
        "player"
    ]

    merge_keys = [
        c for c in preferred_keys
        if c in warehouse.columns and c in detail.columns
    ]

    if not merge_keys:
        return warehouse

    # --------------------------------------------------------
    # Colonne calcolate dal backtesting
    # --------------------------------------------------------
    detail_cols = [
        c for c in [
            "actual_points",
            "actual_wins",
            "actual_matches_played",
            "prediction_error",
            "efficiency_ratio",
            "rounds_won"
        ]
        if c in detail.columns
    ]

    if not detail_cols:
        return warehouse

    detail_small = (
        detail[
            merge_keys + detail_cols
        ]
        .drop_duplicates(
            subset=merge_keys
        )
        .copy()
    )

    merged = warehouse.merge(
        detail_small,
        on=merge_keys,
        how="left",
        suffixes=("", "_bt")
    )

    # --------------------------------------------------------
    # Helper: riempie le colonne vuote originali con i valori _bt
    # --------------------------------------------------------
    def fill_from_backtest(target_col, source_col):

        if source_col not in merged.columns:
            return

        if target_col not in merged.columns:
            merged[target_col] = pd.NA

        merged[target_col] = (
            merged[target_col]
            .replace(
                {
                    "": pd.NA,
                    "None": pd.NA,
                    "none": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA
                }
            )
        )

        merged[target_col] = merged[target_col].combine_first(
            merged[source_col]
        )

    fill_from_backtest("actual_points", "actual_points_bt")
    fill_from_backtest("actual_wins", "actual_wins_bt")
    fill_from_backtest("actual_matches_played", "actual_matches_played_bt")
    fill_from_backtest("prediction_error", "prediction_error_bt")
    fill_from_backtest("efficiency_ratio", "efficiency_ratio_bt")

    # --------------------------------------------------------
    # actual_best_round deriva da rounds_won
    # --------------------------------------------------------
    if "rounds_won" in merged.columns:

        if "actual_best_round" not in merged.columns:
            merged["actual_best_round"] = pd.NA

        merged["actual_best_round"] = (
            merged["actual_best_round"]
            .replace(
                {
                    "": pd.NA,
                    "None": pd.NA,
                    "none": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA
                }
            )
        )

        merged["actual_best_round"] = (
            merged["actual_best_round"]
            .combine_first(
                merged["rounds_won"]
            )
        )

    # --------------------------------------------------------
    # Rimuove colonne tecniche duplicate
    # --------------------------------------------------------
    cols_to_drop = [
        c for c in merged.columns
        if c.endswith("_bt") or c == "rounds_won"
    ]

    merged = merged.drop(
        columns=cols_to_drop,
        errors="ignore"
    )

    return merged

# ------------------------------------------------------------
# Tabs principali
# ------------------------------------------------------------
tab_pred, tab_summary, tab_actual, tab_backtest, tab_calibration = st.tabs(
    [
        "Predictions",
        "Prediction Warehouse",
        "Actual Results",
        "Backtesting",
        "🧪 Calibration Lab"
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

            display_master_df = st.session_state.get(
                "prediction_log_master_enriched",
                master_df
            )

            st.dataframe(
                display_master_df,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Download Warehouse
            # ----------------------------------------------------
            st.download_button(
                "⬇️ Download prediction_warehouse.csv",
                dataframe_to_csv_bytes(
                    display_master_df
                ),
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


FEATURE_COLUMNS = [

    "selected_surface_elo",
    "overall_elo",
    "peak_elo",

    "recent_form",
    "surface_form_60d",
    "same_surface_ratio_60d",

    "fatigue_load",
    "minutes_30d",

    "service_dominance",
    "return_dominance",

    "qualifier_momentum_raw",
    "local_home_raw",

    "value_index",
    "credits",

    "matches_in_db"

]

def build_feature_correlation_report(
    training_df
):

    rows = []

    for feat in FEATURE_COLUMNS:

        if feat not in training_df.columns:
            continue

        try:

            subset = training_df[
                [
                    feat,
                    "actual_points"
                ]
            ].copy()

            subset = subset.apply(
                pd.to_numeric,
                errors="coerce"
            )

            subset = subset.dropna()

            if len(subset) < 10:
                continue

            corr = (
                subset.corr()
                .iloc[0,1]
            )

            rows.append(
                {
                    "feature": feat,
                    "correlation": round(
                        corr,
                        4
                    ),
                    "abs_corr": round(
                        abs(corr),
                        4
                    )
                }
            )

        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_corr",
            ascending=False
        )
        .reset_index(drop=True)
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

        # ----------------------------------------------------
        # Tournament Mapping
        # ----------------------------------------------------
        st.markdown("### Tournament Mapping")

        mapping_df = build_tournament_mapping(
            pred_df,
            actual_df
        )

        st.dataframe(
            mapping_df,
            use_container_width=True,
            hide_index=True
        )

        if "matched_count" in mapping_df.columns:
            unmatched = mapping_df[
                mapping_df["matched_count"] == 0
            ].copy()

            if not unmatched.empty:
                st.warning(
                    "Some prediction tournaments were not matched with TennisMyLife tournament names."
                )

                st.dataframe(
                    unmatched,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success(
                    "All prediction tournaments have at least one TennisMyLife match."
                )

        # ----------------------------------------------------
        # Predicted Players vs Actual Winners
        # ----------------------------------------------------
        st.markdown("### Predicted Players vs Actual Winners")

        player_match_df, unmatched_players_df = build_predicted_players_actual_match(
            pred_df,
            actual_df
        )

        if not player_match_df.empty:

            total_predicted_players = player_match_df[
                "predicted_player"
            ].nunique()

            matched_players = int(
                player_match_df["found_in_actuals"].sum()
            )

            player_match_rate = (
                matched_players / total_predicted_players * 100
                if total_predicted_players > 0
                else 0
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Predicted Players",
                    total_predicted_players
                )

            with p2:
                st.metric(
                    "Found in Actuals",
                    matched_players
                )

            with p3:
                st.metric(
                    "Player Match Rate",
                    f"{player_match_rate:.1f}%"
                )

            st.dataframe(
                player_match_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download predicted_players_actual_match.csv",
                dataframe_to_csv_bytes(player_match_df),
                file_name="predicted_players_actual_match.csv",
                mime="text/csv",
                key="download_predicted_players_actual_match"
            )

            if not unmatched_players_df.empty:
                st.warning(
                    "Some predicted players were not found in actual TennisMyLife winners."
                )

                st.dataframe(
                    unmatched_players_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download unmatched_predicted_players.csv",
                    dataframe_to_csv_bytes(unmatched_players_df),
                    file_name="unmatched_predicted_players.csv",
                    mime="text/csv",
                    key="download_unmatched_predicted_players"
                )

            else:
                st.success(
                    "All predicted players were found in actual TennisMyLife winners."
                )

        else:
            st.info(
                "Player matching is not available. Check prediction data and actual data columns."
            )

        
        # ----------------------------------------------------
        # Prediction vs Actual Global
        # ----------------------------------------------------
        st.markdown(
            "### Prediction vs Actual Global"
        )

        prediction_actual_df = (
            build_prediction_vs_actual_global(
                pred_df,
                actual_df
            )
        )

        if not prediction_actual_df.empty:

            # -----------------------------------
            # KPI Summary
            # -----------------------------------
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Players Compared",
                    len(prediction_actual_df)
                )

            with c2:
                st.metric(
                    "Average Efficiency",
                    round(
                        prediction_actual_df[
                            "efficiency_ratio"
                        ].mean(),
                        2
                    )
                )

            with c3:
                st.metric(
                    "Total Prediction Error",
                    round(
                        prediction_actual_df[
                            "prediction_error"
                        ].sum(),
                        1
                    )
                )

            # -----------------------------------
            # Detail Table
            # -----------------------------------
            st.dataframe(
                prediction_actual_df,
                use_container_width=True,
                hide_index=True
            )

           st.download_button(
            "⬇️ Download prediction_vs_actual_global.csv",
            dataframe_to_csv_bytes(
            prediction_actual_df
            ),
            file_name="prediction_vs_actual_global.csv",
            mime="text/csv",
            key="download_prediction_vs_actual_global"
        )

            # ----------------------------------------------------
            # Prediction vs Actual by Tournament
            # ----------------------------------------------------
st.markdown("### Prediction vs Actual by Tournament")
        st.markdown("### Prediction vs Actual by Tournament")

        tournament_detail_df, tournament_summary_df = (
            build_prediction_vs_actual_tournament(
                pred_df,
                actual_df
            )
        )

        # ----------------------------------------------------
        # Enriched Prediction Warehouse
        # ----------------------------------------------------
        enriched_prediction_warehouse = (
            enrich_prediction_warehouse_with_actuals(
                pred_df,
                tournament_detail_df
            )
        )

        st.session_state[
            "prediction_log_master_enriched"
        ] = enriched_prediction_warehouse

        if not tournament_detail_df.empty:

            # -----------------------------------------------
            # Strategy / Tournament Summary
            # -----------------------------------------------
            st.markdown("#### Strategy Summary")

            st.dataframe(
                tournament_summary_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download prediction_vs_actual_tournament_summary.csv",
                dataframe_to_csv_bytes(
                    tournament_summary_df
                ),
                file_name="prediction_vs_actual_tournament_summary.csv",
                mime="text/csv",
                key="download_prediction_vs_actual_tournament_summary"
            )

            # -----------------------------------------------
            # KPI
            # -----------------------------------------------
            k1, k2, k3 = st.columns(3)

            with k1:
                st.metric(
                    "Tournament Rows",
                    len(tournament_detail_df)
                )

            with k2:
                avg_efficiency = (
                    tournament_summary_df["efficiency_ratio"].mean()
                    if "efficiency_ratio" in tournament_summary_df.columns
                    else 0
                )

                st.metric(
                    "Avg Team Efficiency",
                    round(avg_efficiency, 3)
                )

            with k3:
                total_error = (
                    tournament_summary_df["prediction_error"].sum()
                    if "prediction_error" in tournament_summary_df.columns
                    else 0
                )

                st.metric(
                    "Total Team Error",
                    round(total_error, 2)
                )

            # -----------------------------------------------
            # Enriched Full Prediction Warehouse
            # -----------------------------------------------
            st.markdown("#### Enriched Full Prediction Warehouse")

            if (
                "prediction_log_master_enriched" in st.session_state
                and not st.session_state["prediction_log_master_enriched"].empty
            ):

                enriched_preview_df = st.session_state[
                    "prediction_log_master_enriched"
                ]

                st.dataframe(
                    enriched_preview_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download enriched_prediction_warehouse.csv",
                    dataframe_to_csv_bytes(
                        enriched_preview_df
                    ),
                    file_name="enriched_prediction_warehouse.csv",
                    mime="text/csv",
                    key="download_enriched_prediction_warehouse"
                )

            # -----------------------------------------------
            # Filters
            # -----------------------------------------------
            st.markdown("#### Player Detail")

            tournament_options = (
                ["All Tournaments"]
                + sorted(
                    tournament_detail_df["tournament"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

            selected_bt_tournament = st.selectbox(
                "Filter tournament",
                tournament_options,
                key="bt_tournament_filter"
            )

            strategy_options = (
                ["All Strategies"]
                + sorted(
                    tournament_detail_df["strategy"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

            selected_bt_strategy = st.selectbox(
                "Filter strategy",
                strategy_options,
                key="bt_strategy_filter"
            )

            filtered_tournament_detail = tournament_detail_df.copy()

            if selected_bt_tournament != "All Tournaments":
                filtered_tournament_detail = filtered_tournament_detail[
                    filtered_tournament_detail["tournament"]
                    == selected_bt_tournament
                ].copy()

            if selected_bt_strategy != "All Strategies":
                filtered_tournament_detail = filtered_tournament_detail[
                    filtered_tournament_detail["strategy"]
                    == selected_bt_strategy
                ].copy()

            st.dataframe(
                filtered_tournament_detail,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download prediction_vs_actual_tournament_detail.csv",
                dataframe_to_csv_bytes(
                    tournament_detail_df
                ),
                file_name="prediction_vs_actual_tournament_detail.csv",
                mime="text/csv",
                key="download_prediction_vs_actual_tournament_detail"
            )

        else:
            st.info(
                "Prediction vs Actual by Tournament is not available. Check tournament mapping and required columns."
            )
        
        # ----------------------------------------------------
        # Loaded Data Summary
        # ----------------------------------------------------
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

        st.write("1. Filter TennisMyLife matches by tournament and year")
        st.write("2. Count wins for each predicted player")
        st.write("3. Calculate actual points using wins * 25")
        st.write("4. Compare expected points vs actual points")
        st.write("5. Calculate prediction error and efficiency ratio")

# ------------------------------------------------------------
# TAB 5 — Calibration Lab
# ------------------------------------------------------------
with tab_calibration:

    st.subheader("Weight Calibration Lab")

    if "prediction_log_master_enriched" not in st.session_state:

        st.info("Esegui prima il Backtesting.")

    else:

        training_df = (
            st.session_state["prediction_log_master_enriched"]
            .copy()
        )

        # ----------------------------
        # Clean dataset
        # ----------------------------
        training_df["actual_points"] = pd.to_numeric(
            training_df["actual_points"],
            errors="coerce"
        )

        training_df = training_df[
            training_df["actual_points"].notna()
        ].copy()

        # ----------------------------
        # KPI
        # ----------------------------
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Training Rows",
                len(training_df)
            )

        with c2:
            st.metric(
                "Unique Players",
                training_df["player"].nunique()
            )

        # ----------------------------
        # Correlation
        # ----------------------------
        corr_df = build_feature_correlation_report(training_df)

        st.markdown("### Feature Correlation")

        if corr_df.empty:

            st.warning("Dataset troppo piccolo.")

        else:

            st.dataframe(
                corr_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Top Predictors")

            st.dataframe(
                corr_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download feature_correlation.csv",
                dataframe_to_csv_bytes(corr_df),
                file_name="feature_correlation.csv",
                mime="text/csv",
                key="download_feature_correlation"
            )
