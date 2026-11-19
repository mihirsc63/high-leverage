from __future__ import annotations
import argparse, json, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from pybaseball import statcast

CACHE_DIR = Path(__file__).parent / "cache" / "statcast"
MANIFEST = CACHE_DIR / "_pulled_dates.json"
CHUNK_DAYS = 7
PAUSE_SECONDS = 3
MAX_RETRIES = 3
# Rough MLB window (spring training through the World Series). An empty response for a
# chunk inside this window is suspicious, so those days are not marked as pulled.
SEASON_WINDOW = ((3, 1), (11, 5))


def date_chunks(start: date, end: date, days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    """Return a list of date ranges from start to end, each with a maximum of `days` days."""
    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks

def _missing_runs(days: list[str], pulled: set[str]) -> list[tuple[date, date]]:
    """Group the days not yet pulled into contiguous (start, end) runs."""
    runs: list[tuple[date, date]] = []
    for d in (date.fromisoformat(x) for x in days if x not in pulled):
        if runs and d == runs[-1][1] + timedelta(days=1):
            runs[-1] = (runs[-1][0], d)
        else:
            runs.append((d, d))
    return runs

def _in_season(start: date, end: date) -> bool:
    """True if any day in start..end falls inside SEASON_WINDOW."""
    (sm, sd), (em, ed) = SEASON_WINDOW
    return any(date(d.year, sm, sd) <= d <= date(d.year, em, ed)
               for d in (start + timedelta(i) for i in range((end - start).days + 1)))

def _load_manifest() -> set[str]:
    """Load the manifest file, which tracks the last pulled date for each year."""
    if MANIFEST.exists():
        return set(json.loads(MANIFEST.read_text()))
    return set()

def _save_manifest(pulled: set[str]) -> None:
    """Save the manifest file with the updated pulled dates."""
    MANIFEST.write_text(json.dumps(sorted(pulled), indent=0))

def _days(start: date, end: date) -> list[str]:
    """Return the number of days between start and end, inclusive."""
    return [(start + timedelta(d)).isoformat() for d in range((end - start).days + 1)]

def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    """Fetch a chunk of statcast data from start to end."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return statcast(start_dt=start.isoformat(), end_dt=end.isoformat(),
                            verbose=False, parallel=False)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"  retry {attempt} for {start}..{end} in {wait:.0f}s ({e})")
            time.sleep(wait)

def pull_statcast_range(start: date, end: date, chunk_days: int = CHUNK_DAYS,
                        pause: float = PAUSE_SECONDS, mark_empty: bool = False) -> list[Path]:
    """Pull statcast data for a given date range, in chunks."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pulled = _load_manifest()
    written: list[Path] = []
    hit_network = False
    # never cache today or later: those games may still be incomplete
    end = min(end, date.today() - timedelta(days=1))

    # Only fetch days that are missing, so chunks never overlap existing parquet files
    # (overlapping files would store the same pitches twice).
    missing = _missing_runs(_days(start, end), pulled) if start <= end else []
    if not missing:
        print(f"  {start}..{end} already pulled")
    chunks = [c for run in missing for c in date_chunks(run[0], run[1], chunk_days)]

    for c_start, c_end in chunks:
        if hit_network:
            time.sleep(pause)
        df = _fetch_chunk(c_start, c_end)
        hit_network = True
        if not df.empty:
            path = CACHE_DIR / f"statcast_{c_start}_{c_end}.parquet"
            df.to_parquet(path, index=False)
            written.append(path)
        elif _in_season(c_start, c_end) and not mark_empty:
            # Could be a Savant hiccup: leave unmarked so the next run retries.
            print(f"  WARNING {c_start}..{c_end} returned 0 rows in season; not marking as pulled "
                  f"(rerun with --mark-empty if there really were no games)")
            continue
        pulled.update(_days(c_start, c_end))
        _save_manifest(pulled)
        print(f"  pulled {c_start}..{c_end} ({len(df)} rows)")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull statcast data for a given date range.")
    parser.add_argument("start"); parser.add_argument("end")
    parser.add_argument("--chunk-days", type=int, default=CHUNK_DAYS)
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    parser.add_argument("--mark-empty", action="store_true",
                        help="mark in-season days that return 0 rows as pulled")
    args = parser.parse_args()
    pull_statcast_range(date.fromisoformat(args.start), date.fromisoformat(args.end),
                        args.chunk_days, args.pause, args.mark_empty)