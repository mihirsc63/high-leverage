"""Read-only access to cached data. Every other layer gets its data through these getters."""
from __future__ import annotations

import gzip
import json
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

DATA_ROOT = Path(os.environ.get("HL_DATA_ROOT", Path(__file__).parent / "cache"))
STATCAST_GLOB = DATA_ROOT / "statcast" / "*.parquet"

REGULAR_SEASON = ("R",)
POSTSEASON = ("F", "D", "L", "W")
ALL_MLB_GAMES = REGULAR_SEASON + POSTSEASON

COLUMNS = (
    # game and pitch identity
    "game_pk", "game_year", "game_type", "at_bat_number", "pitch_number",
    # players
    "pitcher", "batter", "player_name", "stand", "p_throws",
    # the pitch itself
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate",
    # location
    "plate_x", "plate_z", "zone", "sz_top", "sz_bot",
    # result
    "description", "events", "type", "launch_speed", "launch_angle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "estimated_woba_using_speedangle", "delta_run_exp",
    # game state
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b", "home_team", "away_team",
    "home_score", "away_score", "bat_score", "fld_score",
)
PITCH_ORDER = "game_date, game_pk, at_bat_number, pitch_number"


@lru_cache(maxsize=1)
def _connection() -> duckdb.DuckDBPyConnection:
    """Open one in-memory DuckDB connection with a `pitches` view over the parquet cache."""
    if not any(STATCAST_GLOB.parent.glob(STATCAST_GLOB.name)):
        raise FileNotFoundError(f"No Statcast parquet files at {STATCAST_GLOB}. Run data.statcast first.")
    con = duckdb.connect(":memory:")
    con.execute(f"""
        CREATE VIEW pitches AS
        SELECT CAST(game_date AS DATE) AS game_date, {", ".join(COLUMNS)}
        FROM read_parquet('{STATCAST_GLOB.as_posix()}', union_by_name = true)
    """)
    return con

def _run(sql: str, params: list) -> pd.DataFrame:
    """Run a parameterized query on its own cursor and return a DataFrame."""
    return _connection().cursor().execute(sql, params).df()

def _select_pitches(condition: str, params: list, season: int | None, before: date | None,
                    game_types: tuple[str, ...]) -> pd.DataFrame:
    """Fetch pitches matching `condition` plus the shared season, date, and game-type filters."""
    clauses = [condition, "list_contains(?, game_type)"]
    values = [*params, list(game_types)]
    if season is not None:
        clauses.append("game_year = ?")
        values.append(season)
    if before is not None:
        clauses.append("game_date < ?")
        values.append(before)
    return _run(f"SELECT * FROM pitches WHERE {' AND '.join(clauses)} ORDER BY {PITCH_ORDER}", values)


def get_pitcher_pitches(pitcher_id: int, season: int | None = None, *, before: date | None = None,
                        game_types: tuple[str, ...] = REGULAR_SEASON) -> pd.DataFrame:
    """Every pitch thrown by a pitcher. Feeds pitch mix, platoon splits, and zones."""
    return _select_pitches("pitcher = ?", [pitcher_id], season, before, game_types)


def get_batter_pitches(batter_id: int, season: int | None = None, *, before: date | None = None,
                       game_types: tuple[str, ...] = REGULAR_SEASON) -> pd.DataFrame:
    """Every pitch seen by a batter. Feeds batter splits and zones."""
    return _select_pitches("batter = ?", [batter_id], season, before, game_types)


def get_matchup_pitches(pitcher_id: int, batter_id: int, *, before: date | None = None,
                        game_types: tuple[str, ...] = ALL_MLB_GAMES) -> pd.DataFrame:
    """Every pitch between one pitcher and one batter across all cached seasons."""
    return _select_pitches("pitcher = ? AND batter = ?", [pitcher_id, batter_id], None, before, game_types)


def get_pitcher_workload(pitcher_id: int, before: date, days: int = 14,
                         game_types: tuple[str, ...] = ALL_MLB_GAMES) -> pd.DataFrame:
    """One row per recent game with pitch count, newest first. Feeds reliever availability."""
    since = before - timedelta(days=days)
    return _run(
        """
        SELECT game_date, game_pk, COUNT(*) AS pitches
        FROM pitches
        WHERE pitcher = ? AND list_contains(?, game_type) AND game_date >= ? AND game_date < ?
        GROUP BY game_date, game_pk
        ORDER BY game_date DESC, game_pk DESC
        """,
        [pitcher_id, list(game_types), since, before],
    )


def get_game_pitches(game_pk: int) -> pd.DataFrame:
    """Every pitch of one game in order. Feeds the eval harness replay."""
    return _run(f"SELECT * FROM pitches WHERE game_pk = ? ORDER BY {PITCH_ORDER}", [game_pk])


MLB_CACHE = DATA_ROOT / "mlb"
PLAYERS_PATH = MLB_CACHE / "players.parquet"


def teams_path(season: int) -> Path:
    return MLB_CACHE / "teams" / f"teams_{season}.parquet"


def schedule_path(season: int) -> Path:
    return MLB_CACHE / "schedules" / f"schedule_{season}.parquet"


def roster_path(team_id: int, season: int, on_date: date | None = None) -> Path:
    label = on_date.isoformat() if on_date else "full"
    return MLB_CACHE / "rosters" / str(season) / f"team_{team_id}_{label}.parquet"


def feed_path(game_pk: int) -> Path:
    return MLB_CACHE / "feeds" / f"{game_pk}.json.gz"


def _read_cached(path: Path, order_by: str) -> pd.DataFrame:
    """Read one cached parquet table, or explain which pull is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is not cached. Run: python -m data.mlb_api season <year>")
    return _run(f"SELECT * FROM read_parquet(?) ORDER BY {order_by}", [path.as_posix()])


def get_teams(season: int) -> pd.DataFrame:
    """All 30 teams with IDs and abbreviations."""
    return _read_cached(teams_path(season), "team_id")


def get_schedule(season: int, *, played_only: bool = True, before: date | None = None,
                 game_types: tuple[str, ...] = ALL_MLB_GAMES) -> pd.DataFrame:
    """One row per unique game. Feeds the eval harness game list."""
    games = _read_cached(schedule_path(season), "official_date, game_pk")
    games = games[games["game_type"].isin(game_types)]
    if played_only:
        games = games[games["played"]]
    if before is not None:
        games = games[games["official_date"] < pd.Timestamp(before)]
    return games.reset_index(drop=True)


def get_roster(team_id: int, season: int, on_date: date | None = None) -> pd.DataFrame:
    """A team's full-season roster, or its active roster on one date if it was pulled."""
    return _read_cached(roster_path(team_id, season, on_date), "player_id")


def get_players() -> pd.DataFrame:
    """Names, positions, and handedness for every cached player."""
    return _read_cached(PLAYERS_PATH, "player_id")


def get_game_feed(game_pk: int) -> dict:
    """The full cached play-by-play feed for one game."""
    path = feed_path(game_pk)
    if not path.exists():
        raise FileNotFoundError(f"{path} is not cached. Run: python -m data.mlb_api feed {game_pk}")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def get_statcast_player_ids() -> list[int]:
    """Every pitcher and batter ID in the Statcast cache."""
    ids = _run("SELECT pitcher AS player_id FROM pitches UNION SELECT batter FROM pitches", [])
    return sorted(ids["player_id"].astype(int).tolist())