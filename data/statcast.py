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


def date_chunks(start: date, end: date, days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    """Return a list of date ranges from start to end, each with a maximum of `days` days."""
    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks

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
                        pause: float = PAUSE_SECONDS) -> list[Path]:
    """Pull statcast data for a given date range, in chunks."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pulled = _load_manifest()
    written: list[Path] = []
    hit_network = False
    # never cache today or later: those games may still be incomplete
    end = min(end, date.today() - timedelta(days=1))

    for c_start, c_end in date_chunks(start, end, chunk_days):
        days = _days(c_start, c_end)
        if all(d in pulled for d in days):
            print(f"  skipping {c_start}..{c_end} (already pulled)")
            continue
        if hit_network:
            time.sleep(pause)
        df = _fetch_chunk(c_start, c_end)
        hit_network = True
        if not df.empty:
            path = CACHE_DIR / f"statcast_{c_start}_{c_end}.parquet"
            df.to_parquet(path, index=False)
            written.append(path)
        pulled.update(days)
        _save_manifest(pulled)
        print(f"  pulled {c_start}..{c_end} ({len(df)} rows)")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull statcast data for a given date range.")
    parser.add_argument("start"); parser.add_argument("end")
    parser.add_argument("--chunk-days", type=int, default=CHUNK_DAYS)
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args()
    pull_statcast_range(date.fromisoformat(args.start), date.fromisoformat(args.end),
                        args.chunk_days, args.pause)