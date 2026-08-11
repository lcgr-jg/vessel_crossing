# Red Sea — W. Coast Saudi bypass trace

Systematic tracking of crude tankers loading on Saudi Arabia’s Red Sea / West Coast after the **20 Jul 2026** Houthi announcement on Bab el Mandeb transit.

## Patterns

| Class | Sequence |
|-------|----------|
| `full_cycle` | W. Coast load → Ain Sukhna discharge → Sidi Kerir reload (→ final dest) |
| `sidi_topup` | W. Coast **partial** load (&lt;65% capacity) → Sidi Kerir top-up (no Ain Sukhna) |
| `ain_sukhna_only` | W. Coast load → Ain Sukhna, **no** Sidi Kerir reload / Suez exit within **7 days** |
| `in_progress` | Legs still missing, or still inside the 7-day window |
| `other` | W. Coast load not aimed at this Ain Sukhna pattern |

## How to run

1. Kernel: repo `.venv` (has `kpler.sdk`).
2. Credentials: `vessel_crossing/.env` → `KPLER_EMAIL` / `KPLER_PASSWORD`.
3. Open and run `red_sea_bypass_trace.ipynb` top-to-bottom.
4. CSVs land in `output/`.

## Knobs

Edit `config.py`:

- `START_DATE`, `AIN_SUKHNA_ONLY_DAYS`
- `W_COAST_ORIGIN_KEYWORDS`, port aliases
- `FORWARD_DAYS` for scheduled / in-transit coverage

In the notebook volume cell:

- `BOOK = "all"` or `"realized_only"` — charts/tables honor this (Scheduled fixtures are hatched when `all`)

## Outputs

- `output/wcoast_universe.csv` — all W. Coast crude loads in window
- `output/journey_chains_detail.csv` — uncollapsed Kpler trade rows (cargo splits kept)
- `output/journey_chains.csv` — collapsed journeys used in charts (one row per IMO×load_date)
- `output/final_destinations.csv` — completed cycles only
- `output/volume_daily.csv` — daily kbbl time series (load / discharge / reload / dump)
- `output/topdown_summary.csv` — Yanbu / Ain / Sidi top-down vs bottom-up comparison
- `output/topdown_yanbu_destinations.csv` / `topdown_sidi_destinations.csv` — dest mixes
- `output/topdown_suez_side_daily.csv` — daily Yanbu ex-Ain + Sidi Kerir stack
- `output/topdown_suez_side_by_region_daily.csv` — same book stacked by USA/Europe/Asia/Other
- `output/topdown_suez_side_by_country.csv` — country breakdown of that book
- `output/topdown_suez_side_vessels_by_region.csv` — vessel-level list by region (Kpler cross-check)
