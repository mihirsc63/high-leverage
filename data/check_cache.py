"""Sanity-check the cached Statcast season: game counts, duplicates, team totals, thin days.

Usage: python -m data.check_cache
"""
from __future__ import annotations
import duckdb
from data.statcast import CACHE_DIR

SRC = f"read_parquet('{CACHE_DIR}/*.parquet', union_by_name=true)"
GAMES_PER_TEAM = 162
THIN_DAY_GAMES = 5


def main() -> None:
    con = duckdb.connect()

    print("Pitches and games by game_type "
          "(R=regular, F=wild card, D=division, L=LCS, W=World Series, S=spring):")
    print(con.sql(f"""select game_type, count(*) pitches, count(distinct game_pk) games
                      from {SRC} group by 1 order by games desc"""))

    dupes = con.sql(f"""select count(*) from (
        select game_pk, at_bat_number, pitch_number from {SRC}
        group by all having count(*) > 1)""").fetchall()[0][0]
    print(f"Duplicate pitches on (game_pk, at_bat_number, pitch_number): {dupes}"
          + ("  <-- PROBLEM" if dupes else "  OK"))

    off = con.sql(f"""select team, count(distinct game_pk) g from (
        select home_team team, game_pk from {SRC} where game_type='R'
        union all
        select away_team, game_pk from {SRC} where game_type='R')
        group by 1 having g <> {GAMES_PER_TEAM} order by 1""")
    if off.fetchall():
        print(f"\nTeams without {GAMES_PER_TEAM} regular-season games "
              "(two teams at 161 = likely a rainout never made up; check their schedules):")
        print(off)
    else:
        print(f"All teams have {GAMES_PER_TEAM} regular-season games  OK")

    print(f"\nRegular-season days with fewer than {THIN_DAY_GAMES} games "
          "(expect Opening Night, All-Star break edges; anything else may be a missed pull):")
    print(con.sql(f"""select game_date, count(distinct game_pk) g from {SRC}
                      where game_type='R' group by 1 having g < {THIN_DAY_GAMES} order by 1"""))


if __name__ == "__main__":
    main()
