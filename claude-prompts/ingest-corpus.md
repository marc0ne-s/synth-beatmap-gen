# SynthRiders: Ingest Real .synth Corpus — Claude Code Prompt

## Context

SynthRiders is an AI beatmap generation project. We have 4,413 lines of existing FastAPI + React code in `/Volumes/Second-Brain-1/Home/Desktop/SynthGen/` that was trained on synthetic data only (57% accuracy). We now have the password for real user-made maps.

The password is: `hC2*wE5R*qQzv@a!`

## Working Scripts (Already Tested)

- `/Volumes/Second-Brain-1/AI/Synth/scripts/synth_password_extractor.py` — extracts password from OBB
- `/Volumes/Second-Brain-1/AI/Synth/scripts/synth_decryptor.py` — reads/writes/batch-decrypts .synth files

## Source Data

- **Raw .synth files:** `/Volumes/Second-Brain-1/Meta Quest/Synth/` (exactly 10,918 files)
- **Working directory:** `/Volumes/Second-Brain-1/AI/Synth/`
- **Target dataset directory:** `/Volumes/Second-Brain-1/AI/Synth/dataset/`

## Immediate Objective

Ingest the 10,918 real .synth files into a structured, machine-learning-ready dataset.

### Phase 1: File Operations
1. Create `/Volumes/Second-Brain-1/AI/Synth/dataset/raw/` if it doesn't exist
2. Copy (or symlink) all `.synth` files from the source directory to `dataset/raw/`
3. Verify count matches 10,918

### Phase 2: Decrypt & Extract
1. Use `synth_decryptor.py` to batch-decrypt all files to `dataset/decrypted/`
2. Each `.synth` is a ZIP containing:
   - `beatmap.meta.bin` (encrypted metadata, format partially known)
   - `beatmap.data` (note timing + position data — this is the gold)
   - `cover.jpg` (album art)
   - `info.json` (song metadata: BPM, title, artist, difficulty)
3. Extract the raw ZIP contents to `dataset/extracted/<uuid>/`
4. Generate a master index CSV/JSON at `dataset/index.json` with fields:
   - `file_id` (UUID from filename)
   - `song_title`, `artist`, `bpm`, `difficulty` (from info.json)
   - `note_count` (from beatmap.data)
   - `file_paths` (relative paths to extracted content)

### Phase 3: Beatmap Data Parsing
1. Inspect the structure of `beatmap.data` — it's binary Note data
2. Document the binary format in `dataset/specs/beatmap.data.md`
3. Write a parser (Python) that converts `beatmap.data` into structured JSON/Parquet:
   - Each note: `{time, lane, type, position, velocity}` (fields TBD by inspection)

### Phase 4: Validation & Stats
1. Report how many files decrypted successfully vs failed
2. Report basic corpus stats: total notes, avg notes per map, BPM distribution, difficulty distribution
3. Save validation report to `dataset/reports/ingestion-report-YYYY-MM-DD.md`

## Constraints
- Do NOT modify originals in `/Volumes/Second-Brain-1/Meta Quest/Synth/`
- Keep logs of any failed decryptions with filenames
- Use streaming/chunked operations — 10,918 files could be several GB
- Progress bars or periodic status prints appreciated

## Deliverables
- [ ] `dataset/raw/` populated with 10,918 files
- [ ] `dataset/decrypted/` with processed outputs
- [ ] `dataset/extracted/` with unpacked ZIP contents
- [ ] `dataset/index.json` master index
- [ ] `dataset/specs/beatmap.data.md` format documentation
- [ ] Parser script at `scripts/parse_beatmap_data.py`
- [ ] Ingestion report with stats

## Notes
- HerBB (the local agent) has already extracted the password and written the decryptor scripts. If you hit issues with the scripts, check them first before rewriting.
- If the batch decrypt is slow, consider parallelising with `concurrent.futures` or `multiprocessing`.
- The goal of this entire project is to train a model on real human-made beatmaps to generate AI beatmaps. This ingestion step is the foundation.
