# IPL Strategy Engine

Streamlit app for IPL pre-match strategy, player-vs-player analytics, venue intelligence, and phase analysis using Cricsheet ball-by-ball data.

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Rebuild the SQLite database from the official IPL JSON zip:

```bash
python -m cricket_analytics.ingest --download
```

or

```bash
python -m cricket_analytics.ingest --force-download
```

3. Launch the app:

```bash
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
```

## Data sources

- Cricsheet JSON format: https://cricsheet.org/format/json/
- Cricsheet downloads: https://cricsheet.org/downloads/

## Notes

- The official IPL zip is stored at `data/raw/ipl_json.zip`.
- Rebuilds clear the existing SQLite tables and ingest the full IPL dataset again.
- Spin/pace and left/right-arm weakness analysis becomes richer if you later enrich `players` with bowling style metadata.
