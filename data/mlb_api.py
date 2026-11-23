"""Network client for the MLB Stats API. Pulls once and writes to the cache. Agents never import this."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from data.queries import (ALL_MLB_GAMES, PLAYERS_PATH, feed_path, get_players, get_statcast_player_ids,
                          roster_path, schedule_path, teams_path)

BASE_URL = "https://statsapi.mlb.com"
MLB_SPORT_ID = 1
PAUSE_SECONDS = 1
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
PEOPLE_BATCH_SIZE = 100
PLAYED_STATES = {"Final", "Game Over", "Completed Early"}

_session = requests.Session()

def _get_json(path: str, params: dict | None = None) -> dict:
    """GET one API path politely, retrying only failures that may be temporary."""
    error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(PAUSE_SECONDS)
        try:
            response = _session.get(BASE_URL + path, params=params, timeout=TIMEOUT_SECONDS)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response.json()
            error = f"HTTP {response.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            error = str(e)
        if attempt < MAX_RETRIES:
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"  retry {attempt} for {path} in {wait}s ({error})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {path} after {MAX_RETRIES} attempts: ({error})")

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write through a temp file so a crash can never leave a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temp, index=False)
    temp.replace(path)


def pull_teams(season: int) -> Path:
    """Cache the 30 MLB teams for a season."""
    path = teams_path(season)
    if path.exists():
        print(f"  skipping teams {season} (cached)")
        return path
    teams = _get_json("/api/v1/teams", {"sportId": MLB_SPORT_ID, "season": season})["teams"]
    rows = [{"team_id": t["id"], "name": t["name"], "abbreviation": t["abbreviation"]} for t in teams]
    _write_parquet(pd.DataFrame(rows), path)
    print(f"  pulled {len(rows)} teams")
    return path


def pull_schedule(season: int) -> Path:
    """Cache one row per unique game, keeping the played version of postponed games."""
    path = schedule_path(season)
    if path.exists():
        print(f"  skipping schedule {season} (cached)")
        return path
    params = {"sportId": MLB_SPORT_ID, "season": season, "gameType": ",".join(ALL_MLB_GAMES)}
    data = _get_json("/api/v1/schedule", params)
    games = [game for day in data["dates"] for game in day["games"]]
    if any(game["status"]["abstractGameState"] != "Final" for game in games):
        raise RuntimeError(f"Season {season} has unfinished games, so its schedule can't be cached yet.")

    rows = [{
        "game_pk": game["gamePk"],
        "official_date": game["officialDate"],
        "game_type": game["gameType"],
        "status": game["status"]["detailedState"],
        "played": game["status"]["detailedState"] in PLAYED_STATES,
        "home_team_id": game["teams"]["home"]["team"]["id"],
        "home_team": game["teams"]["home"]["team"]["name"],
        "away_team_id": game["teams"]["away"]["team"]["id"],
        "away_team": game["teams"]["away"]["team"]["name"],
        "home_score": game["teams"]["home"].get("score"),
        "away_score": game["teams"]["away"].get("score"),
        "double_header": game.get("doubleHeader"),
        "game_number": game.get("gameNumber"),
    } for game in games]

    schedule = pd.DataFrame(rows)
    schedule["official_date"] = pd.to_datetime(schedule["official_date"])
    schedule = (schedule.sort_values("played")
                .drop_duplicates("game_pk", keep="last")
                .sort_values(["official_date", "game_pk"]))
    _write_parquet(schedule, path)
    print(f"  pulled {len(schedule)} games ({int(schedule['played'].sum())} played)")
    return path


def pull_roster(team_id: int, season: int, on_date: date | None = None) -> Path:
    """Cache a team's full-season roster, or its active roster on a past date."""
    if on_date is not None and on_date >= date.today():
        raise ValueError("Only past rosters can be cached.")
    path = roster_path(team_id, season, on_date)
    if path.exists():
        return path
    params = ({"rosterType": "active", "date": on_date.isoformat()} if on_date
              else {"rosterType": "fullSeason", "season": season})
    roster = _get_json(f"/api/v1/teams/{team_id}/roster", params)["roster"]
    rows = [{
        "team_id": team_id,
        "player_id": player["person"]["id"],
        "full_name": player["person"]["fullName"],
        "position": player["position"]["abbreviation"],
        "status": player["status"]["description"],
    } for player in roster]
    _write_parquet(pd.DataFrame(rows), path)
    return path


def pull_players(player_ids: list[int]) -> Path:
    """Cache names and handedness, fetching only players not already stored."""
    existing = get_players() if PLAYERS_PATH.exists() else pd.DataFrame()
    known = set(existing["player_id"]) if not existing.empty else set()
    missing = sorted(set(player_ids) - known)
    if not missing:
        print("  skipping players (cached)")
        return PLAYERS_PATH

    rows = []
    for start in range(0, len(missing), PEOPLE_BATCH_SIZE):
        batch = missing[start:start + PEOPLE_BATCH_SIZE]
        people = _get_json("/api/v1/people", {"personIds": ",".join(map(str, batch))})["people"]
        rows += [{
            "player_id": person["id"],
            "full_name": person["fullName"],
            "position": person["primaryPosition"]["abbreviation"],
            "bat_side": person["batSide"]["code"],
            "pitch_hand": person["pitchHand"]["code"],
        } for person in people]

    players = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    _write_parquet(players, PLAYERS_PATH)
    print(f"  pulled {len(rows)} players ({len(players)} total)")
    return PLAYERS_PATH


def pull_game_feed(game_pk: int) -> Path:
    """Cache the full play-by-play feed of one finished game."""
    path = feed_path(game_pk)
    if path.exists():
        print(f"  skipping feed {game_pk} (cached)")
        return path
    feed = _get_json(f"/api/v1.1/game/{game_pk}/feed/live")
    status = feed["gameData"]["status"]["detailedState"]
    if status not in PLAYED_STATES:
        raise RuntimeError(f"Game {game_pk} is '{status}', so its feed can't be cached.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as file:
        json.dump(feed, file)
    temp.replace(path)
    print(f"  pulled feed {game_pk}")
    return path


def pull_season(season: int) -> None:
    """Pull everything for a season except game feeds, which are pulled per game."""
    pull_teams(season)
    pull_schedule(season)
    teams = pd.read_parquet(teams_path(season))
    for team_id in teams["team_id"]:
        pull_roster(int(team_id), season)
    print(f"  rosters ready for {len(teams)} teams")
    pull_players(get_statcast_player_ids())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull MLB Stats API data into the local cache.")
    commands = parser.add_subparsers(dest="command", required=True)
    season_parser = commands.add_parser("season", help="pull teams, schedule, rosters, and players")
    season_parser.add_argument("year", type=int)
    feed_parser = commands.add_parser("feed", help="pull one or more game feeds")
    feed_parser.add_argument("game_pks", type=int, nargs="+")
    args = parser.parse_args()

    if args.command == "season":
        pull_season(args.year)
    else:
        for game_pk in args.game_pks:
            pull_game_feed(game_pk)
