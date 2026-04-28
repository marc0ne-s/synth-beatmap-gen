# SynthRiders AI — Phase 2: Feature Engineering, Baseline Model, & Dashboard Automation
## Autonomous Agent Prompt
**Date:** 2026-04-25
**Project root:** `/Volumes/Second-Brain-1/AI/Synth/` (ML pipeline)
**Dashboard root:** `/Volumes/Second-Brain-1/AI/synthriders/dashboard/` (Player dashboard)
**Agent:** Claude Code (via `claude --acp --stdio`)

---

## PART A: ML TRAINING PIPELINE (Priority 1)

### 1. What Just Happened (State of Play)

Corpus ingestion is **COMPLETE**. Summary:

| Metric | Value |
|--------|-------|
| .synth files found | 8,301 |
| Decrypted | 8,300 (99.99%) |
| Parsed beatmaps | 4,638 |
| Total notes | 5,202,229 |
| Avg notes/map | 1,122 |

**Difficulty distribution:** Master 44.7% (2,075 maps), Expert 37.8% (1,751), Hard 24.8% (1,152).

**The .synth format** is a ZIP archive containing `beatmap.meta.bin`, which despite the extension is **UTF-8 JSON with BOM**. It stores an entire beatmap — metadata, all difficulties, and base64-encoded artwork — in a single file. Only ~55.9% of extracted archives had this file; the rest are legacy/incomplete.

**Deliverables already created:**
- `dataset/extracted/<uuid>/` — decrypted ZIP contents
- `dataset/parsed/<uuid>.json` — structured beatmap data
- `dataset/parsed/index.json` — master parsed index
- `dataset/reports/ingestion-report-2026-04-25.md` — full report
- `scripts/batch_extract.py` — batch decryptor
- `scripts/parse_beatmap_data.py` — beatmap-to-JSON parser
- `dataset/specs/beatmap.data.md` — format documentation

---

### 2. Parsed Data Schema (CRITICAL)

Each `<uuid>.json` has this structure:

```json
{
  "status": "success",
  "path": ".../beatmap.meta.bin",
  "metadata": {
    "name": "Supernova",
    "author": "Within Temptation",
    "bpm": 127.0,
    "offset": 0.0,
    "beatmapper": "Technical",
    "editor_version": "1.9.7-10",
    "production_mode": true,
    "modified_time": 1607372250
  },
  "difficulties": {
    "Normal": { "note_count": 1016, "notes": [...] }
  },
  "stats": { "total_notes": 1016, ... }
}
```

**Note structure** (each element in `difficulties.<name>.notes`):

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Timestamp in **milliseconds** |
| `x` | float | Horizontal position (~ -0.54 to +0.54) |
| `y` | float | Vertical position (~ -0.41 to +0.41) |
| `z` | float | Depth / timing axis. **NOT world Z**. Use `time` as canonical temporal axis. |
| `type` | int | **0 = Right hand, 1 = Left hand** |
| `direction` | int | Rail entry angle (mostly 0 in corpus) |
| `combo_id` | int | Combo grouping ID (-1 = no combo) |
| `id` | string | Unique note ID |
| `segments` | list[list[float]]\|null | **Rails / slides** — list of [x, y, z] points. Present on ~25-30% of notes. |

**Key observations:**
- x/y coordinates cluster around a quantized grid (~6-8 horizontal slots, ~4-5 vertical).
- Rails are critical for flow and difficulty prediction.
- The `z` coordinate is a playback parameter, not independent time.

---

### 3. Track A: Feature Engineering (MUST DO FIRST)

Create a module that converts each parsed beatmap into model-ready tensors.

For each beatmap, extract:

1. **Temporal grid:** Discretize into fixed-width bins (e.g., 20ms or beat-synchronous).
2. **Note occupancy tensor:** Per-frame, per-hand, a spatial grid or continuous (x,y) representation.
3. **Rail tensor:** Per-frame, binary or path-encoded feature indicating active rails.
4. **Beat-aligned metadata:** BPM, time signature (if inferable), downbeat positions.
5. **Difficulty label:** Normal / Hard / Expert / Master (categorical).

**Design decisions to make and document:**
- Frame resolution: fixed-time vs. beat-synchronous (try both)
- Spatial encoding: continuous vs. discretized grid
- Rail encoding: rasterize per-frame, Bézier parametrize, or treat as additional note positions

Create:
- `scripts/feature_engineering.py` or `src/features/` package
- `dataset/features/` directory (precomputed features per UUID)
- A `SynthBeatmapDataset` PyTorch Dataset class

---

### 4. Track B: Audio Analysis Pipeline (CAN RUN IN PARALLEL)

Build a pipeline that will consume raw audio files. We don't have matching audio files yet.

Create:
- `src/audio/` using **librosa** (or torchaudio):
  - Mel spectrogram (80-128 bins)
  - Chroma (12-bin pitch class)
  - Tempogram / onset envelope
  - Beat-synchronous aggregation
  - CQT as alternative to STFT
- `scripts/extract_audio_features.py` — standalone script for inference
- Document expected format in `dataset/specs/audio-features.md`

**Without audio files:**
- Design the audio encoder architecture
- Create a **synthetic audio feature generator** for the baseline overfit test
- Validate the map decoder independently

---

### 5. Track C: Baseline Model & Overfit Test

Build a minimal, overfittable model to prove the pipeline works.

**Recommended architecture:**

```
Audio Features (T, F) 
  -> Audio Encoder (Conv1D stack or small Transformer) 
  -> Latent sequence (T, D)
  -> Note Decoder (LSTM or Transformer with causal masking) 
  -> Per-frame note predictions:
       - Note presence (binary or multi-class per hand)
       - X, Y coordinates (regression)
       - Rail flag + path params
```

**Train on a tiny subset first:**
- Pick 100 maps (or even 10)
- Use synthetic/random audio features for now
- Train until training loss drops to near-zero (overfit)
- **Success criteria:** >90% note recall, <50ms timing error on training set

Create:
- `src/models/` package
- `src/training/train_baseline.py` — training loop
- `scripts/overfit_baseline.sh` — one-command run
- Save checkpoints to `models/checkpoints/`

---

### 6. Infrastructure Decision

There is an **existing project** at `/Volumes/Second-Brain-1/Home/Desktop/SynthGen/` (4,413 LOC FastAPI + React). It was synthetic-only (57% accuracy). 

**Recommendation:** Start fresh for the ML core. Read the existing code to steal good ideas, then build a new `src/` layout under `/Volumes/Second-Brain-1/AI/Synth/`.

---

### 7. ML Deliverables Checklist

- [ ] Feature engineering module (Track A)
- [ ] Audio feature extractor design + standalone script (Track B)
- [ ] Baseline model architecture (Track C)
- [ ] Overfit test on 10-100 maps with synthetic audio input
- [ ] Training loss curves and sample predictions (visualized)
- [ ] Updated project documentation (README.md, spec docs)
- [ ] `requirements.txt` or `pyproject.toml`
- [ ] One-command training script

---

## PART B: RIDER DASHBOARD AUTOMATION (Priority 2)

### 8. Dashboard Source of Truth

**Dashboard directory:** `/Volumes/Second-Brain-1/AI/synthriders/dashboard/`
**Frontend:** `index.html` (HTML/CSS/JS with Chart.js)
**Data files:** `data/merged_stats.json`, `data/quest-saves/*.bin`

### 9. Current Data Sources

**A. Quest Save Files (local, pulled via ADB):**
- Path on Quest: `/sdcard/Android/data/com.kluge.SynthRiders/files/songstats.bin`
- Format: JSON
- Contains: `statsBySong` with per-song scores, accuracy, completions, perfects, difficulty, last played
- Also: `played.bin` (list of played song names), `favorites.bin`, `settings.bin`

**B. Synthriderz.com API (remote, public endpoints discovered):**

```
GET https://synthriderz.com/api/leaderboards        → general song list
GET https://synthriderz.com/api/scores?profile=<9109954> → individual scores (NOT full leaderboard)
GET https://synthriderz.com/api/beatmaps?author=<marc0ne2081> → user's authored maps
GET https://synthriderz.com/api/playlists?user=<9109954> → user's playlists
GET https://synthriderz.com/api/rankings?leaderboard_profile_id=<9109954> → ranked data
```

**Important discovery:** The `/api/scores?profile=9109954` endpoint does NOT return marc0ne2081's scores—it returns various random players' scores. The user's **Synthriderz username is NOT the same as a search term** on the scores endpoint. This means we cannot currently fetch marc0ne2081's individual leaderboard positions via the public API.

**However**, the profile webpage (`https://synthriderz.com/profile/marc0ne2081/leaderboards?profile=9109954`) renders leaderboard data client-side. When the site is stable, scraping the page content gives:
- Rank (#1,164)
- Ranked Scores (100)
- Ranked Accuracy (53.4%)
- Recent plays with accuracy, score, song name, artist, type (Official/Custom)

### 10. Dashboard Automation Tasks

**Task B1: Build a Python scraper/loader**
Create `scripts/sync_synthriderz.py` that:
1. Queries `https://synthriderz.com/api/beatmaps?author=marc0ne2081` for authored maps
2. Queries `https://synthriderz.com/api/playlists?user=9109954` for playlists
3. Uses Playwright (`mcp_playwright_init_browser`) to load the profile page and extract rendered leaderboard data (as fallback when API returns wrong data)
4. Merges with Quest save data into `data/merged_stats.json`
5. Runs on a schedule or on-demand

**Task B2: Enhance the dashboard**
- Add a "Refresh Data" button to the dashboard that calls `sync_synthriderz.py`
- Add a time-series view (skill progression over time using `LastTimePlayed` from Quest data)
- Add a heatmap of most-played songs by week
- Add accuracy trend lines (Quest good_hit vs synthriderz ranked accuracy)
- Add a "Compare to Global" section (pull global averages from `/api/rankings`)

**Task B3: Quest sync automation**
- Create `scripts/sync_quest.py` that checks if Quest is connected via ADB and auto-pulls save files
- Add a timestamp check: only pull if files are newer than last sync

### 11. Dashboard Deliverables Checklist

- [ ] `scripts/sync_synthriderz.py` — API scraper + Playwright fallback
- [ ] `scripts/sync_quest.py` — ADB auto-sync for Quest saves
- [ ] Time-series and accuracy trend charts
- [ ] Auto-refresh button in `index.html`
- [ ] Global comparison stats (rank distribution, average accuracy by difficulty)
- [ ] Documentation of all API endpoints discovered (`docs/synthriderz-api.md`)

---

## 12. Technical Constraints

- **Hardware:** Mac mini M4 Pro, 64GB RAM, Apple Silicon only (MLX or PyTorch MPS), no CUDA
- **Storage:** ~2TB external, keep files tidy (Parquet/compressed NPZ for features)
- **Python:** 3.14.3 via Homebrew (`/opt/homebrew/bin/python3`)
- **No audio files yet:** Design for this; generate synthetic audio features for baseline test
- **Keep it simple:** Research spike, optimize for experimentation speed
- **ADB available:** android-platform-tools installed via Homebrew

---

## 13. How to Report Progress

Update `/Volumes/Second-Brain-1/AI/Synth/AGENT_LOG.md` (create if needed) after each milestone:

```markdown
## YYYY-MM-DD HH:MM — [Project] [Milestone Name]
- What was done
- Key decisions made
- Blockers (if any)
- Next step
```

Also update `/Volumes/Second-Brain-1/AI/synthriders/dashboard/AGENT_LOG.md` for dashboard work.

---

## 14. GOAL

**End Phase 2 with:**
1. A working ML training loop that can overfit 100 beatmaps from synthetic audio input
2. A self-updating rider dashboard that pulls from both Quest saves and Synthriderz.com
3. Clear documentation of all API endpoints and data pipelines

**Start by:**
1. Reading the existing codebase and parsed data
2. Reading the existing SynthGen project (steal good ideas, ignore synthetic-only pipeline)
3. Building the feature engineering module first
4. Then running a small overfit test
5. After ML baseline works, build the dashboard scrapers

**Launch when ready. Report back on AGENT_LOG.md with progress.**
