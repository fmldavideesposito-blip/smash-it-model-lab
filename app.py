import io
import base64
import requests
import re
import unicodedata
import pandas as pd
import streamlit as st


GITHUB_OWNER = st.secrets["GITHUB_OWNER"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]

def load_csv_from_github(path):

    try:

        url = (
            f"https://raw.githubusercontent.com/"
            f"{GITHUB_OWNER}/"
            f"{GITHUB_REPO}/main/"
            f"{path}"
        )

        return pd.read_csv(
            url,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )

    except Exception as e:

        st.warning(
            f"Unable to load {path} from GitHub"
        )

        st.exception(e)

        return pd.DataFrame()

# ------------------------------------------------------------
# Config pagina
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smash IT Model Lab by Davide Esposito",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Smash IT Model Lab by Davide Esposito")
st.caption("Prediction Backtesting & Model Calibration")

st.success("GitHub Secrets Loaded")

st.write(
    "Owner:",
    GITHUB_OWNER
)

st.write(
    "Repo:",
    GITHUB_REPO
)

st.write(
    "Token presente:",
    len(GITHUB_TOKEN) > 20
)

test_df = load_csv_from_github(
    "data/prediction_warehouse_master.csv"
)

st.write(
    "Rows in GitHub warehouse:",
    len(test_df)
)

# ------------------------------------------------------------
# Costanti TennisMyLife
# ------------------------------------------------------------
TML_DATA_FILES_API = "https://stats.tennismylife.org/api/data-files"
POINTS_PER_WIN = 25

# ------------------------------------------------------------
# Utility CSV
# ------------------------------------------------------------

def upload_csv_to_github(
    df,
    path,
    commit_message
):

    csv_content = df.to_csv(
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False
    )

    api_url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/"
        f"{GITHUB_REPO}/contents/"
        f"{path}"
    )

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    sha = None

    existing = requests.get(
        api_url,
        headers=headers
    )

    if existing.status_code == 200:
        sha = existing.json()["sha"]

    payload = {
        "message": commit_message,
        "content": base64.b64encode(
            csv_content.encode("utf-8-sig")
        ).decode("utf-8")
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(
        api_url,
        headers=headers,
        json=payload
    )
    
    response.raise_for_status()

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

from pathlib import Path

DATA_DIR = Path("data")

DATA_DIR.mkdir(exist_ok=True)

PRED_MASTER_FILE = (
    DATA_DIR / "prediction_warehouse_master.csv"
)

ACTUAL_MASTER_FILE = (
    DATA_DIR / "actual_results_master.csv"
)

def load_prediction_master():

    try:

        return load_csv_from_github(
            "data/prediction_warehouse_master.csv"
        )

    except Exception as e:

        st.error(
            f"Errore caricamento warehouse GitHub: {e}"
        )

        return pd.DataFrame()

def save_prediction_master(df):

    upload_csv_to_github(
        df=df,
        path="data/prediction_warehouse_master.csv",
        commit_message=(
            f"Update Warehouse "
            f"{pd.Timestamp.now()}"
        )
    )

def load_actual_master():

    if ACTUAL_MASTER_FILE.exists():

        try:
            return pd.read_csv(
                ACTUAL_MASTER_FILE,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        except Exception:
            pass

    return pd.DataFrame()

def save_actual_master(df):

    df.to_csv(
        ACTUAL_MASTER_FILE,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False
    )


PRED_KEYS = [
    "run_id",
    "run_timestamp",
    "tournament",
    "year",
    "strategy",
    "player_norm"
]


ACTUAL_KEYS = [
    "tourney_name",
    "tourney_date",
    "winner_name",
    "loser_name",
    "round"
]

def merge_actual_results(
    master_df,
    new_df
):

    merged = pd.concat(
        [
            master_df,
            new_df
        ],
        ignore_index=True
    )

    merged = merged.drop_duplicates(
        subset=ACTUAL_KEYS,
        keep="last"
    )

    return merged

def get_existing_tournaments(
    master_df
):

    if master_df.empty:
        return []

    if "tournament" not in master_df.columns:
        return []

    return sorted(
        master_df["tournament"]
        .dropna()
        .unique()
        .tolist()
    )

def merge_prediction_log(
    master_df,
    new_df,
    replace_tournament=False
):

    if master_df.empty:
        return new_df.copy()

    tournament = (
        new_df["tournament"].iloc[0]
    )

    year = (
        new_df["year"].iloc[0]
    )

    if replace_tournament:

        master_df = master_df[
            ~(
                (master_df["tournament"] == tournament)
                &
                (master_df["year"] == year)
            )
        ].copy()

    merged = pd.concat(
        [
            master_df,
            new_df
        ],
        ignore_index=True
    )

    merged = merged.drop_duplicates(
        subset=PRED_KEYS,
        keep="last"
    )

    return merged

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

    pred_norm = pred_df.copy()

    pred_norm["player_norm"] = (
    pred_norm["player"]
        .apply(normalize_player_name)
    )

    prediction_summary = (
        pred_norm
        .groupby(
            ["player_norm"],
           dropna=False
        )
        .agg(
            player=("player", "first"),
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

def normalize_text_key(value):
    """
    Crea una chiave robusta:
    - minuscolo
    - rimozione accenti
    - rimozione apostrofi, trattini, spazi, simboli
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    text = text.replace("&", "and")

    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text
    )

    return text


TOURNAMENT_CANONICAL_MAP = {
    # --------------------------------------------------------
    # Slam
    # --------------------------------------------------------
    "australianopen": "australianopen",
    "rolandgarros": "rolandgarros",
    "frenchopen": "rolandgarros",
    "wimbledon": "wimbledon",
    "usopen": "usopen",

    # --------------------------------------------------------
    # Masters 1000
    # --------------------------------------------------------
    "indianwells": "indianwells",
    "indianwellsmasters": "indianwells",

    "miami": "miami",
    "miamimasters": "miami",

    "montecarlo": "montecarlo",
    "montecarlomasters": "montecarlo",

    "madrid": "madrid",
    "madridmasters": "madrid",

    "roma": "rome",
    "rome": "rome",
    "romemasters": "rome",
    "internazionalibnlditalia": "rome",

    "cincinnati": "cincinnati",
    "cincinnatimasters": "cincinnati",

    "shanghai": "shanghai",
    "shanghaimasters": "shanghai",

    "parigibercy": "paris",
    "paris": "paris",
    "parismasters": "paris",

    "torontomontreal": "canada",
    "toronto": "canada",
    "torontomasters": "canada",
    "montreal": "canada",
    "montrealmasters": "canada",
    "canadianmasters": "canada",

    # --------------------------------------------------------
    # ATP Finals
    # --------------------------------------------------------
    "nittoatpfinals": "atpfinals",
    "atpfinals": "atpfinals",
    "tourfinals": "atpfinals",

    # --------------------------------------------------------
    # ATP 500 / 250 - nomi italiani vs TennisMyLife
    # --------------------------------------------------------
    "amburgo": "hamburg",
    "hamburg": "hamburg",

    "barcellona": "barcelona",
    "barcelona": "barcelona",

    "basilea": "basel",
    "basel": "basel",

    "ginevra": "geneva",
    "geneva": "geneva",

    "lione": "lyon",
    "lyon": "lyon",

    "monaco": "munich",
    "munich": "munich",

    "pechino": "beijing",
    "beijing": "beijing",

    "stoccolma": "stockholm",
    "stockholm": "stockholm",

    "queens": "queens",
    "queensclub": "queens",
    "london": "queens",

    "shertogenbosch": "hertogenbosch",
    "hertogenbosch": "hertogenbosch",
    "sherogenbosch": "hertogenbosch",

    # --------------------------------------------------------
    # Tornei che normalmente coincidono già
    # --------------------------------------------------------
    "acapulco": "acapulco",
    "adelaide": "adelaide",
    "almaty": "almaty",
    "auckland": "auckland",
    "bastad": "bastad",
    "brisbane": "brisbane",
    "bruxelles": "brussels",
    "brussels": "brussels",
    "bucharest": "bucharest",
    "buenosaires": "buenosaires",
    "chengdu": "chengdu",
    "dallas": "dallas",
    "delraybeach": "delraybeach",
    "doha": "doha",
    "dubai": "dubai",
    "eastbourne": "eastbourne",
    "estoril": "estoril",
    "gstaad": "gstaad",
    "halle": "halle",
    "hangzhou": "hangzhou",
    "hongkong": "hongkong",
    "houston": "houston",
    "kitzbuhel": "kitzbuhel",
    "loscabos": "loscabos",
    "mallorca": "mallorca",
    "marrakech": "marrakech",
    "montpellier": "montpellier",
    "riodejaneiro": "riodejaneiro",
    "rotterdam": "rotterdam",
    "santiago": "santiago",
    "stuttgart": "stuttgart",
    "tokyo": "tokyo",
    "umag": "umag",
    "vienna": "vienna",
    "washington": "washington",
    "winstonsalem": "winstonsalem",
}


def normalize_tournament_name(name):
    """
    Normalizza il nome torneo usando una chiave canonica comune
    tra Optimizer e TennisMyLife.
    """

    key = normalize_text_key(name)

    return TOURNAMENT_CANONICAL_MAP.get(
        key,
        key
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
                    "tournaments_won_matches": 0,
                    "surfaces": "",
                    "rounds_won": "",
                }
            )

    match_df = pd.DataFrame(rows)

    unmatched_df = match_df[
        match_df["found_in_actuals"] == False
    ].copy()

    match_df["tournaments_won_matches"] = pd.to_numeric(
        match_df["tournaments_won_matches"],
        errors="coerce"
    ).fillna(0).astype(int)
    
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

    Importante:
    - sovrascrive gli actual precedenti quando il backtesting produce un valore;
    - mantiene actual_points = 0 quando il giocatore ha davvero perso senza vittorie;
    - aggiunge actual_matches_in_tournament per distinguere tornei completati
      da tornei non ancora presenti negli actual TennisMyLife.
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
            "actual_losses",
            "actual_matches_played",
            "actual_matches_in_tournament",
            "prediction_error",
            "efficiency_ratio",
            "rounds_won"
        ]
        if c in detail.columns
    ]

    if not detail_cols:
        return warehouse

    numeric_detail_cols = [
        "actual_points",
        "actual_wins",
        "actual_losses",
        "actual_matches_played",
        "actual_matches_in_tournament",
        "prediction_error",
        "efficiency_ratio"
    ]

    for col in numeric_detail_cols:
        if col in detail.columns:
            detail[col] = pd.to_numeric(
                detail[col],
                errors="coerce"
            )

    detail_small = (
        detail[
            merge_keys + detail_cols
        ]
        .drop_duplicates(
            subset=merge_keys,
            keep="last"
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
    # Helper: sovrascrive con il valore da backtesting
    # --------------------------------------------------------
    def overwrite_from_backtest(target_col, source_col):

        if source_col not in merged.columns:
            return

        if target_col not in merged.columns:
            merged[target_col] = pd.NA

        mask = merged[source_col].notna()

        merged.loc[
            mask,
            target_col
        ] = merged.loc[
            mask,
            source_col
        ]

    overwrite_from_backtest("actual_points", "actual_points_bt")
    overwrite_from_backtest("actual_wins", "actual_wins_bt")
    overwrite_from_backtest("actual_losses", "actual_losses_bt")
    overwrite_from_backtest("actual_matches_played", "actual_matches_played_bt")
    overwrite_from_backtest("actual_matches_in_tournament", "actual_matches_in_tournament_bt")
    overwrite_from_backtest("prediction_error", "prediction_error_bt")
    overwrite_from_backtest("efficiency_ratio", "efficiency_ratio_bt")

    # --------------------------------------------------------
    # actual_best_round deriva da rounds_won
    # --------------------------------------------------------
    if "rounds_won" in merged.columns:

        if "actual_best_round" not in merged.columns:
            merged["actual_best_round"] = pd.NA

        rounds_mask = (
            merged["rounds_won"]
            .notna()
            &
            (
                merged["rounds_won"]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        merged.loc[
            rounds_mask,
            "actual_best_round"
        ] = merged.loc[
            rounds_mask,
            "rounds_won"
        ]

    # --------------------------------------------------------
    # Pulizia numerica finale
    # --------------------------------------------------------
    numeric_cols = [
        "actual_points",
        "actual_wins",
        "actual_losses",
        "actual_matches_played",
        "actual_matches_in_tournament",
        "prediction_error",
        "efficiency_ratio"
    ]

    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(
                merged[col],
                errors="coerce"
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

def build_dream_team(
    tournament_df: pd.DataFrame,
    budget=100,
    team_size=8
):
    return pd.DataFrame(), 0

def optimize_team_by_score(
    pool_df,
    score_col,
    budget=100,
    team_size=8
):
    """
    Ottimizza una squadra massimizzando score_col
    rispettando budget e team_size.

    Usa crediti decimali convertiti in decimi:
    13.3 -> 133
    8.2  -> 82
    budget 100 -> 1000
    """

    required_cols = [
        "player",
        "credits",
        score_col
    ]

    missing_cols = [
        c for c in required_cols
        if c not in pool_df.columns
    ]

    if missing_cols:
        return pd.DataFrame(), 0, 0

    work_df = pool_df.copy()

    work_df["credits"] = pd.to_numeric(
        work_df["credits"],
        errors="coerce"
    )

    work_df[score_col] = pd.to_numeric(
        work_df[score_col],
        errors="coerce"
    )

    work_df = work_df.dropna(
        subset=[
            "credits",
            score_col
        ]
    ).copy()

    if len(work_df) < team_size:
        return pd.DataFrame(), 0, 0

    work_df = work_df.reset_index(drop=True)

    budget_int = int(
        round(
            budget * 10
        )
    )

    players = []

    for i, row in work_df.iterrows():

        players.append(
            {
                "idx": i,
                "player": row["player"],
                "credits": float(row["credits"]),
                "cost_int": int(
                    round(
                        float(row["credits"]) * 10
                    )
                ),
                "score": float(row[score_col])
            }
        )

    dp = {
        (0, 0): (
            0.0,
            ()
        )
    }

    for p in players:

        nd = dict(dp)

        for (spent, count), val in dp.items():

            if count >= team_size:
                continue

            new_spent = spent + p["cost_int"]

            if new_spent > budget_int:
                continue

            key = (
                new_spent,
                count + 1
            )

            new_score = val[0] + p["score"]

            new_idxs = val[1] + (
                p["idx"],
            )

            candidate = (
                new_score,
                new_idxs
            )

            if (
                key not in nd
                or
                candidate[0] > nd[key][0]
            ):
                nd[key] = candidate

        dp = nd

    best_score = -1
    best_idxs = ()
    best_spent = 0

    for (spent, count), val in dp.items():

        if count == team_size:

            if val[0] > best_score:

                best_score = val[0]
                best_idxs = val[1]
                best_spent = spent

    if not best_idxs:
        return pd.DataFrame(), 0, 0

    team_df = work_df.iloc[
        list(best_idxs)
    ].copy()

    team_df = team_df.sort_values(
        score_col,
        ascending=False
    ).reset_index(drop=True)

    return (
        team_df,
        round(best_score, 2),
        round(best_spent / 10, 1)
    )


def optimize_expected_team(
    pool_df,
    budget=100,
    team_size=8
):
    team_df, total_points, total_credits = optimize_team_by_score(
        pool_df=pool_df,
        score_col="expected_points",
        budget=budget,
        team_size=team_size
    )

    return team_df, total_points

    players = []

    for _, row in pool_df.iterrows():

        players.append(
            (
                row["player"],
                float(row["credits"]),
                float(row["expected_points"])
            )
        )

    dp = {
        (0, 0): (
            0,
            ()
        )
    }

    for i, (_, credits, points) in enumerate(players):

        nd = dict(dp)

        cost = int(round(credits))

        for (spent, count), val in dp.items():

            if count >= team_size:
                continue

            new_spent = spent + cost

            if new_spent > budget:
                continue

            key = (
                new_spent,
                count + 1
            )

            score = val[0] + points

            idxs = val[1] + (i,)

            candidate = (
                score,
                idxs
            )

            if (
                key not in nd
                or
                candidate[0] > nd[key][0]
            ):
                nd[key] = candidate

        dp = nd

    best_score = -1
    best_idxs = ()

    for (spent, count), val in dp.items():

        if count == team_size:

            if val[0] > best_score:

                best_score = val[0]
                best_idxs = val[1]

    if not best_idxs:

        return pd.DataFrame(), 0

    team = pool_df.iloc[
        list(best_idxs)
    ].copy()

    return (
        team.sort_values(
            "expected_points",
            ascending=False
        ),
        round(best_score, 2)
    )

def build_actual_points_for_pool(
    pool_df,
    actual_df,
    actual_tournament,
    actual_year=None,
    points_per_win=25
):
    """
    Arricchisce il pool del ranking_completo.csv con:
    - actual_wins
    - actual_points
    - actual_matches_in_tournament

    usando i risultati reali TennisMyLife.
    """

    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    if actual_df is None or actual_df.empty:
        return pool_df.copy()

    if "winner_name" not in actual_df.columns:
        return pool_df.copy()

    if "tourney_name" not in actual_df.columns:
        return pool_df.copy()

    pool = pool_df.copy()
    actual = actual_df.copy()

    pool["player_norm"] = (
        pool["player"]
        .apply(normalize_player_name)
    )

    actual = actual[
        actual["tourney_name"]
        .astype(str)
        == str(actual_tournament)
    ].copy()

    if actual_year is not None and actual_year != "All Years":

        if "source_year" in actual.columns:

            actual["actual_year"] = pd.to_numeric(
                actual["source_year"],
                errors="coerce"
            )

        elif "tourney_date" in actual.columns:

            actual["actual_year"] = pd.to_numeric(
                actual["tourney_date"]
                .astype(str)
                .str.slice(0, 4),
                errors="coerce"
            )

        else:

            actual["actual_year"] = pd.NA

        actual = actual[
            actual["actual_year"] == int(actual_year)
        ].copy()

    if actual.empty:

        pool["actual_wins"] = 0
        pool["actual_points"] = 0
        pool["actual_matches_in_tournament"] = 0

        return pool

    actual["winner_norm"] = (
        actual["winner_name"]
        .apply(normalize_player_name)
    )

    wins_df = (
        actual
        .groupby(
            "winner_norm",
            dropna=False
        )
        .agg(
            actual_wins=("winner_name", "count")
        )
        .reset_index()
    )

    wins_df["actual_points"] = (
        wins_df["actual_wins"]
        * points_per_win
    )

    pool = pool.merge(
        wins_df,
        left_on="player_norm",
        right_on="winner_norm",
        how="left"
    )

    pool["actual_wins"] = (
        pool["actual_wins"]
        .fillna(0)
        .astype(int)
    )

    pool["actual_points"] = (
        pool["actual_points"]
        .fillna(0)
        .astype(float)
    )

    pool["actual_matches_in_tournament"] = len(actual)

    pool = pool.drop(
        columns=[
            "winner_norm"
        ],
        errors="ignore"
    )

    return pool


# ------------------------------------------------------------
# Tabs principali
# ------------------------------------------------------------
tab_pred, tab_summary, tab_actual, tab_backtest, tab_calibration, tab_dream, tab_ideal = st.tabs(
    [
        "Predictions",
        "Prediction Warehouse",
        "Actual Results",
        "Backtesting",
        "🧪 Calibration Lab",
        "🏆 Dream Team Lab",
        "🏆 Ideal Team Backtest",
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

    # ----------------------------------------------------
    # Auto-load warehouse persistente
    # ----------------------------------------------------
    if "prediction_log_master" not in st.session_state:

        master_df = load_prediction_master()

        if not master_df.empty:

            st.session_state[
                "prediction_log_master"
            ] = master_df
    
    uploaded_logs = st.file_uploader(
        "Upload one or more prediction logs",
        type=["csv"],
        accept_multiple_files=True,
        key="prediction_warehouse"
    )

    # ----------------------------------------------------
    # Visualizza warehouse già esistente
    # ----------------------------------------------------
    if (
        "prediction_log_master" in st.session_state
        and uploaded_logs is None
    ):

        master_df = st.session_state[
            "prediction_log_master"
        ]

        with st.expander("DEBUG Warehouse Tournaments"):

            st.write(
                sorted(
                    master_df["tournament"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

        

        st.success(
            f"{len(master_df)} rows loaded from GitHub."
        )

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

    if uploaded_logs:

        all_logs = []

        master_df = load_prediction_master()

        if (
            not master_df.empty
            and "player_norm" not in master_df.columns
        ):
            master_df["player_norm"] = (
                master_df["player"]
                .apply(normalize_player_name)
            )
        for f in uploaded_logs:

            try:
                df = read_prediction_log(f)

                st.write(
                    "FILE:",
                    f.name,
                    "TOURNAMENT:",
                    df["tournament"].iloc[0]
                )

                df["player_norm"] = (
                    df["player"]
                    .apply(normalize_player_name)
                )

                tournament_name = (
                 df["tournament"].iloc[0]
                )

                year = (
                 df["year"].iloc[0]
                )

                already_exists = False

                if not master_df.empty:

                 already_exists = len(
                     master_df[
                          (master_df["tournament"] == tournament_name)
                            &
                            (master_df["year"] == year)
                        ]
                    ) > 0

                if already_exists:

                 action = st.radio(
                    f"{tournament_name} {year} già presente",
                    [
                        "Keep Existing",
                        "Replace Existing",
                        "Append Anyway"
                    ],
                  key=f"dup_{tournament_name}"
                    )

                else:

                    action = "Append Anyway"

                if action == "Keep Existing":

                    st.info(
                        "Master non modificato."
                    )

                else:

                    master_df = merge_prediction_log(
                        master_df,
                        df,
                        replace_tournament=(
                        action == "Replace Existing"
                        )
                    )

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

            except Exception as e:

                st.error(
                    f"Errore caricamento {f.name}"
                )

                st.exception(e)

        save_prediction_master(
            master_df
        )

        st.write(
            "TORNEI MASTER IN MEMORIA:"
            )

        st.write(
            sorted(
                master_df["tournament"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        st.success(
            f"Warehouse salvato su GitHub "
            f"({len(master_df)} rows)"
        )
        
        if all_logs:

            st.session_state[
                "prediction_log_master"
            ] = master_df

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

# ----------------------------------------------------
# Auto-load actual results persistenti
# ----------------------------------------------------
if "actual_results" not in st.session_state:

    actual_master = load_actual_master()

    if not actual_master.empty:

        st.session_state[
            "actual_results"
        ] = actual_master

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

                    if not loaded_actuals:

                        st.warning(
                        "No TennisMyLife seasons were loaded."
                        )

                        st.stop()
                    
                    actual_df = pd.concat(
                        loaded_actuals,
                        ignore_index=True
                    )

                    actual_master = load_actual_master()

                    actual_master = merge_actual_results(
                        actual_master,
                        actual_df
                    )

                    save_actual_master(
                        actual_master
                    )

                    st.session_state[
                        "actual_results"
                    ] = actual_master

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

            actual_master = load_actual_master()

            actual_master = merge_actual_results(
                actual_master,
                actual_df
            )

            save_actual_master(
                actual_master
            )

            st.session_state[
                "actual_results"
            ] = actual_master

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

    "elo_win_probability",
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

    "matches_in_db",
    "expected_points",

]

def elo_to_win_probability(
    elo_value,
    reference_elo=2000
):
    """
    Trasforma Elo in probabilità di vittoria
    rispetto ad un giocatore di riferimento.

    2000 è una buona baseline ATP.
    """

    try:

        elo_diff = (
            float(elo_value)
            - float(reference_elo)
        )

        return 1 / (
            1 + 10 ** (-elo_diff / 400)
        )

    except Exception:
        return None

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

        with st.expander("DEBUG Tournament Names"):

            st.write("Prediction tournaments:")

            st.write(
                sorted(
                    pred_df["tournament"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

            st.write("Actual tournaments:")

            st.write(
                sorted(
                    actual_df["tourney_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

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
        if "actual_matches_in_tournament" in training_df.columns:

            training_df["actual_matches_in_tournament"] = pd.to_numeric(
                training_df["actual_matches_in_tournament"],
                errors="coerce"
            ).fillna(0)

            training_df = training_df[
                training_df["actual_matches_in_tournament"] > 0
            ].copy()

        else:

            st.warning(
                "La colonna actual_matches_in_tournament non è disponibile. "
                "Esegui prima il Backtesting con la nuova funzione di enrichment."
            )

            st.stop()

        if "actual_points" not in training_df.columns:

            st.warning(
                "La colonna actual_points non è disponibile. "
                "Esegui prima il Backtesting."
            )

            st.stop()

        training_df["actual_points"] = pd.to_numeric(
            training_df["actual_points"],
            errors="coerce"
        )

        training_df = training_df[
            training_df["actual_points"].notna()
        ].copy()

        if "selected_surface_elo" in training_df.columns:

            training_df["elo_win_probability"] = (
                training_df["selected_surface_elo"]
                .apply(
                    lambda x: elo_to_win_probability(
                        x,
                        reference_elo=2000
                    )
                )
            )

        if len(training_df) == 0:

            st.warning(
                "No calibrated rows available. Run backtesting first."
            )

            st.stop()

        # ----------------------------
        # KPI before strategy filter
        # ----------------------------
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Training Rows before strategy filter",
                len(training_df)
            )

        with c2:
            st.metric(
                "Unique Players before strategy filter",
                training_df["player"].nunique()
                if "player" in training_df.columns
                else 0
            )

        if "tournament" in training_df.columns:

            completed_tournaments = (
                training_df["tournament"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            st.caption(
                "Tournaments included in calibration: "
                + ", ".join(sorted(completed_tournaments))
            )

        # ----------------------------
        # Strategy filter
        # ----------------------------
        if "strategy" in training_df.columns:

            training_df = training_df[
                training_df["strategy"]
                == "1. Optimized Team"
            ].copy()

            st.caption(
                "Calibration is currently based only on: 1. Optimized Team"
            )

        if len(training_df) == 0:

            st.warning(
                "No rows available after strategy filter."
            )

            st.stop()

        # ----------------------------
        # KPI after strategy filter
        # ----------------------------
        c3, c4 = st.columns(2)

        with c3:
            st.metric(
                "Training Rows after strategy filter",
                len(training_df)
            )

        with c4:
            st.metric(
                "Unique Players after strategy filter",
                training_df["player"].nunique()
                if "player" in training_df.columns
                else 0
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

# ------------------------------------------------------------
# TAB 6 — Dream Team Lab
# ------------------------------------------------------------
with tab_dream:

    st.subheader("🏆 Dream Team Lab")

    if "prediction_log_master_enriched" not in st.session_state:

        st.info(
            "Run Backtesting first."
        )

    else:

        warehouse = (
            st.session_state[
                "prediction_log_master_enriched"
            ]
        )

        st.success(
            f"{len(warehouse)} rows available."
        )

        st.dataframe(
            warehouse.head(50),
            use_container_width=True,
            hide_index=True
        )

        run_options = (
            warehouse["run_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_run = st.selectbox(
            "Select Run",
            run_options
        )

        strategy_filter = st.selectbox(
            "Strategy",
            [
                "All Strategies",
                "1. Optimized Team",
                "2. Aggressive Alternative",
                "3. Conservative Alternative"
            ],
            key="dream_strategy_filter"
        )


        run_df = warehouse[
            warehouse["run_id"].astype(str)
            == str(selected_run)
        ].copy()

        if strategy_filter != "All Strategies":

            run_df = run_df[
                run_df["strategy"]
                == strategy_filter
            ].copy()

        st.write("Rows in run:", len(run_df))

        st.write(
            run_df[
                ["player","strategy"]
            ]
            .sort_values(
                ["player","strategy"]
            )       
        )

        run_df["player_norm"] = (
            run_df["player"]
            .apply(normalize_player_name)
            )

        available_players_df = (
            run_df
            .sort_values(
                "expected_points",
                ascending=False
            )
            .drop_duplicates(
                subset=["player_norm"]
            )
        )

        st.write(
            "Players available:",
            len(available_players_df)
        )

        st.dataframe(
            available_players_df[
                [
                    "player",
                    "credits",
                    "actual_points",
                    "strategy"
                ]
            ],
            use_container_width=True
        )

        budget = int(
            run_df["budget"].iloc[0]
            )

        team_size = int(
                run_df["team_size"].iloc[0]
            )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
        "Budget",
        budget
    )

        with c2:
            st.metric(
                "Team Size",
                team_size
            )

        required_cols = [
            "player",
            "credits",
            "actual_points"
        ]

        missing_cols = [
            c for c in required_cols
            if c not in run_df.columns
        ]

        if missing_cols:

            st.error(
                f"Missing columns: {missing_cols}"
            )

            st.stop()

# ------------------------------------------------------------
# TAB 7 — Ideal Team Backtest
# ------------------------------------------------------------

with tab_ideal:

    st.subheader("🏆 Ideal Team Backtest")

    ranking_file = st.file_uploader(
        "Upload ranking_completo.csv",
        type=["csv"],
        key="ideal_ranking"
    )

    if ranking_file:

        ranking_df = pd.read_csv(
            ranking_file,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )

        st.session_state["ranking_df"] = ranking_df

        st.success(
            f"{len(ranking_df)} players loaded"
        )

        with st.expander("DEBUG Columns"):
            st.write(ranking_df.columns.tolist())

        st.dataframe(
            ranking_df.head(20)
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Players",
                len(ranking_df)
            )

        with c2:
            st.metric(
                "Best Expected Points",
                round(
                    ranking_df["expected_points_v13"].max(),
                    1
                )
            )

        with c3:
            st.metric(
                "Avg Credits",
                round(
                    pd.to_numeric(
                        ranking_df["Smash IT Credits CW N"],
                        errors="coerce"
                    ).mean(),
                    1
                )
            )

        required_cols = [
        "Player",
        "Smash IT Credits CW N",
        "expected_points_v13",
        "rank_v13"
]       

        missing = [
            c for c in required_cols
            if c not in ranking_df.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

            st.stop()

        ideal_pool = ranking_df[
            [
                "Player",
                 "Smash IT Credits CW N",
                "expected_points_v13",
                "rank_v13"
            ]
        ].copy()

        ideal_pool = ideal_pool.rename(
            columns={
                "Player": "player",
                "Smash IT Credits CW N": "credits",
                "expected_points_v13": "expected_points"
                }
        )

        ideal_pool["credits"] = pd.to_numeric(
            ideal_pool["credits"],
            errors="coerce"
        )

        ideal_pool["expected_points"] = pd.to_numeric(
            ideal_pool["expected_points"],
            errors="coerce"
        )

        ideal_pool["credits"] = ideal_pool["credits"].astype(float)
        ideal_pool["expected_points"] = ideal_pool["expected_points"].astype(float)

        if "prediction_log_master" in st.session_state:

            warehouse = st.session_state[
                "prediction_log_master"
            ]

            available_runs = sorted(
                warehouse["run_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_run = st.selectbox(
                "Select Prediction Run",
                available_runs,
                key="ideal_run_selector"
            )

            run_df = warehouse[
            warehouse["run_id"].astype(str)
                == str(selected_run)
            ].copy()

            budget = int(
                pd.to_numeric(
                    run_df["budget"],
                    errors="coerce"
                ).iloc[0]
            )

            team_size = int(
                pd.to_numeric(
                    run_df["team_size"],
                    errors="coerce"
                ).iloc[0]
            )

            st.success(
                f"Run loaded | Budget={budget} | Team Size={team_size}"
            )

        else:

            st.warning(
                "Prediction Warehouse not loaded."
            )

            st.stop()
        
        budget = 100
        team_size = 8
        
        
        ideal_pool = ideal_pool.dropna(
            subset=[
                "credits",
                "expected_points"
            ]
        ).copy()

        if len(ideal_pool) < team_size:

            st.error(
                f"Only {len(ideal_pool)} players available."
            )
        
            st.stop()

        ranking_df = ranking_df.sort_values(
            "rank_v13"
        )

        ideal_pool = ideal_pool.sort_values(
            "rank_v13"
        ).reset_index(drop=True)

        st.markdown(
            "### Top Predicted Players"
        )

        st.dataframe(
            ranking_df[
                [
                    "Player",
                    "Smash IT Credits CW N",
                    "expected_points_v13",
                    "rank_v13"
                ]
            ]
            .head(20),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Ideal Pool")

        st.dataframe(
            ideal_pool.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### Optimization Parameters"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Budget",
                budget
            )

        with c2:
            st.metric(
                "Team Size",
                team_size
        )

        st.write(
            "Total pool expected points:",
            round(
                ideal_pool["expected_points"].sum(),
                2
            )
        )

        st.write(
            "Total pool credits:",
            round(
                ideal_pool["credits"].sum(),
                2
            )
        )

        ideal_team_df, ideal_points = (
            optimize_expected_team(
                ideal_pool,
                budget=budget,
                team_size=team_size
            )
        )

        with st.expander("DEBUG Solver"):

            st.write(
                ideal_team_df[
                    [
                        "player",
                        "credits",
                        "expected_points"
                    ]
                ]
            )

            st.write(
                "Credits:",
                ideal_team_df["credits"].sum()
            )

            st.write(
                "Expected:",
                ideal_team_df["expected_points"].sum()
            )

            st.write(
                ideal_pool.sort_values(
                    "expected_points",
                    ascending=False
                )
                [
                    [
                        "player",
                        "credits",
                        "expected_points",
                        "rank_v13"
                    ]
                ]
                .head(20)
            )

            st.write(
                "Top 20 expected points pool"
            )

            st.write(
                ideal_pool.sort_values(
                    "expected_points",
                    ascending=False
                )
                [
                    [
                        "player",
                        "credits",
                        "expected_points"
                    ]
                ]
                .head(20)
            )

        if ideal_team_df.empty:

            st.error(
                "Unable to generate ideal team."
            )

        else:

            st.success(
                f"Ideal team generated ({len(ideal_team_df)} players)"
            )

            total_credits = (
                ideal_team_df["credits"]
                .sum()
            )

            c1, c2, c3 = st.columns(3)

            st.markdown(
            "### Ideal Team"
            )

            st.write(
                "Credits total:",
                ideal_team_df["credits"].sum()
            )

            st.write(
                "Expected total:",
                ideal_team_df["expected_points"].sum()
            )

            st.dataframe(
            ideal_team_df[
                [
                    "player",
                    "credits",
                    "expected_points",
                    "rank_v13"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

                    # ----------------------------------------------------
        # TRUE IDEAL TEAM BACKTEST - usando actual_points
        # ----------------------------------------------------
        st.markdown(
            "### True Ideal Team Backtest"
        )

        if "actual_results" not in st.session_state:

            st.info(
                "Load Actual Results first to calculate the true ideal team."
            )

        else:

            actual_df = st.session_state["actual_results"]

            if "tourney_name" not in actual_df.columns:

                st.warning(
                    "Actual results do not contain tourney_name."
                )

            else:

                actual_tournament_options = sorted(
                    actual_df["tourney_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_actual_tournament_for_ideal = st.selectbox(
                    "Select actual tournament for Ideal Team Backtest",
                    actual_tournament_options,
                    key="ideal_actual_tournament"
                )

                actual_year_options = [
                    "All Years"
                ]

                if "source_year" in actual_df.columns:

                    actual_year_options += sorted(
                        pd.to_numeric(
                            actual_df["source_year"],
                            errors="coerce"
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )

                elif "tourney_date" in actual_df.columns:

                    actual_year_options += sorted(
                        pd.to_numeric(
                            actual_df["tourney_date"]
                            .astype(str)
                            .str.slice(0, 4),
                            errors="coerce"
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )

                selected_actual_year_for_ideal = st.selectbox(
                    "Select actual year",
                    actual_year_options,
                    key="ideal_actual_year"
                )

                actual_pool = build_actual_points_for_pool(
                    pool_df=ideal_pool,
                    actual_df=actual_df,
                    actual_tournament=selected_actual_tournament_for_ideal,
                    actual_year=selected_actual_year_for_ideal,
                    points_per_win=POINTS_PER_WIN
                )

                st.markdown(
                    "#### Actual Pool"
                )

                st.write(
                    actual_pool.sort_values(
                    "actual_points",
                    ascending=False
                    )
                )

                st.write(
                    actual_pool[
                        actual_pool["actual_points"] > 0
                ]
                    .sort_values(
                        "actual_points",
                        ascending=False
                    )
                )

                st.dataframe(
                    actual_pool[
                        [
                            "player",
                            "credits",
                            "expected_points",
                            "actual_wins",
                            "actual_points",
                            "rank_v13"
                        ]
                    ].sort_values(
                        "actual_points",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                actual_ideal_team_df, actual_ideal_points, actual_ideal_credits = (
                    optimize_team_by_score(
                        pool_df=actual_pool,
                        score_col="actual_points",
                        budget=budget,
                        team_size=team_size
                    )
                )

                if actual_ideal_team_df.empty:

                    st.warning(
                        "Unable to generate true ideal team from actual results."
                    )

                else:

                    st.success(
                        f"True ideal team generated ({len(actual_ideal_team_df)} players)"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Actual Ideal Players",
                            len(actual_ideal_team_df)
                        )

                    with c2:
                        st.metric(
                            "Actual Ideal Credits",
                            round(
                                actual_ideal_credits,
                                1
                            )
                        )

                    with c3:
                        st.metric(
                            "Actual Ideal Points",
                            round(
                                actual_ideal_points,
                                1
                            )
                        )

                    st.markdown(
                        "#### True Ideal Team"
                    )

                    st.dataframe(
                        actual_ideal_team_df[
                            [
                                "player",
                                "credits",
                                "expected_points",
                                "actual_wins",
                                "actual_points",
                                "rank_v13"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    # ------------------------------------------------
                    # Gap Analysis
                    # ------------------------------------------------
                    expected_team_with_actuals = actual_pool[
                        actual_pool["player"].isin(
                            ideal_team_df["player"].tolist()
                        )
                    ].copy()

                    expected_team_actual_points = (
                        expected_team_with_actuals["actual_points"]
                        .sum()
                    )

                    gap_points = (
                        actual_ideal_points
                        - expected_team_actual_points
                    )

                    capture_rate = (
                        expected_team_actual_points
                        / actual_ideal_points
                        * 100
                        if actual_ideal_points > 0
                        else 0
                    )

                    st.markdown(
                        "#### Gap Analysis"
                    )

                    g1, g2, g3 = st.columns(3)

                    with g1:
                        st.metric(
                            "Expected Team Actual Points",
                            round(
                                expected_team_actual_points,
                                1
                            )
                        )

                    with g2:
                        st.metric(
                            "Gap vs True Ideal",
                            round(
                                gap_points,
                                1
                            )
                        )

                    with g3:
                        st.metric(
                            "Capture Rate",
                            f"{capture_rate:.1f}%"
                        )

                    overlap_players = sorted(
                        set(
                            ideal_team_df["player"]
                            .astype(str)
                            .tolist()
                        )
                        &
                        set(
                            actual_ideal_team_df["player"]
                            .astype(str)
                            .tolist()
                        )
                    )

                    st.write(
                        "Overlapping players:",
                        overlap_players
                    )

            with c1:
                st.metric(
                    "Players",
                    len(ideal_team_df)
                )

            with c2:
                st.metric(
                    "Credits Used",
                    round(total_credits, 1)
                )

            with c3:
                st.metric(
                    "Expected Points",
                    round(ideal_points, 1)
                )
