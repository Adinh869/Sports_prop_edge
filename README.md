# Sports Prop Edge

Local Python app for traditional sports DFS / pick'em prop analysis (NBA, KBO, NFL).

Same core idea as `esports_prop_edge`, but **this is a separate project** — own repo, data, and dashboard. No shared code paths with esports.

**Not** an auto-betting bot. No wager placement.

## Live data feeds

| Sport | Package | How |
|-------|---------|-----|
| NBA | `nba_api` | Player game logs from stats.nba.com |
| NFL | `nfl_data_py` | Weekly player stats from nflverse |
| KBO | **MyKBO** (recommended) | [mykbostats.com](https://mykbostats.com) via [Parse API](https://parse.bot) — set `PARSE_API_KEY` |
| KBO | CSV or Statiz | Fallback: local CSV export or Statiz `?s=` id |

```bat
pip install nba_api nfl_data_py lxml html5lib
python tools/fetch_history.py --sport NBA --player "Jaylen Brown" --out data/live/nba_brown.csv
python tools/fetch_history.py --sport NFL --player "Patrick Mahomes" --out data/live/nfl_mahomes.csv
set PARSE_API_KEY=your_parse_key
python tools/fetch_history.py --sport KBO --player "Lee Jung-hoo" --kbo-source mykbo --out data/live/kbo_lee.csv
python tools/fetch_history.py --sport KBO --player "Lee Jung-hoo" --csv-path data/kbo/lee.csv --kbo-source csv --out data/live/kbo_lee.csv
```

Use the **Fetch Data** tab in Streamlit for the same workflow.

## Daily sync (run every day)

Props need **fresh game logs** after each slate. Run once per day (morning or after games):

```bat
set PARSE_API_KEY=your_key
run_daily_sync.bat
```

Or with tonight's PrizePicks export:

```bat
run_daily_sync.bat --props data\props\tonight_props.csv
```

**What it does**

| Sport | Daily behavior |
|-------|----------------|
| **KBO** | MyKBO: 1 schedule call + 1 call per **new** final game (cached — not re-fetched) |
| **NBA** | Refreshes season game log per watchlist/props player |
| **NFL** | Refreshes weekly stats per player (in season) |

Outputs: `data/live/history_merged.csv` (use in Streamlit: **Use daily-synced live history**)

**Windows Task Scheduler**: trigger `run_daily_sync.bat` daily at 8am (KBO) and 9am ET (NBA).

Edit `data/config/watchlist.csv` for your core player pool; drop `data/props/tonight_props.csv` for slate-specific players.

## Quick start

```bat
cd sports_prop_edge
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or double-click `run_app.bat` on Windows.

## CLI

```bat
set PYTHONPATH=src
python -m sports_prop_edge.cli --props data/sample/sample_props_all_sports.csv --history data/sample/sample_history_all_sports.csv --build-cards
```

## Supported sports & markets

| Sport | `game_title` | Markets |
|-------|--------------|---------|
| NBA | `NBA` | points, rebounds, assists, threes, steals, blocks, turnovers, pra, fantasy_points |
| KBO / MLB | `KBO` or `MLB` | hits, runs, rbis, strikeouts, total_bases, walks, stolen_bases, singles, doubles |
| NFL | `NFL` | passing_yards, rushing_yards, receiving_yards, receptions, passing_tds, rushing_tds, receiving_tds |

## Projection logic

- **NBA**: per-minute rate × `expected_minutes`
- **KBO/MLB**: per-plate-appearance rate × `expected_plate_appearances`
- **NFL**: per-game rate × `expected_games`

Adjustments: `opponent_adjustment`, `pace_adjustment`, `home_adjustment`, `weather_adjustment`

## CSV formats

Props: `site, game_title, event_time, player, team, opponent, market, line, side`

History: `date, game_title, player, team, opponent` + stat columns (`minutes`, `plate_appearances`, `points`, `hits`, etc.)

## Project structure

```text
sports_prop_edge/
  app/streamlit_app.py
  data/sample/
  src/sports_prop_edge/
    data/loaders.py
    models/projections.py
    models/distributions.py
    strategy/scoring.py
    strategy/card_builder.py
    strategy/payouts.py
    db/tracker.py
  tests/
```
