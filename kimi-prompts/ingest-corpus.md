# SynthRiders: Ingest Real .synth Corpus — Execution Prompt
## For: Kimi K2.6 Cloud (Current Session)

**DO NOT just plan. Execute immediately.**

## Context

SynthRiders is an AI beatmap generation project. We have 4,413 lines of existing FastAPI + React code in `/Volumes/Second-Brain-1/Home/Desktop/SynthGen/` that was trained on synthetic data only (57% accuracy). We now have the password for real user-made maps.

## Password

`hC2*wE5R*qQzv@a!`

## Working Scripts (Already Tested)

- `/Volumes/Second-Brain-1/AI/projects/synth-gen/scripts/synth_password_extractor.py` — extracts password from OBB
- `/Volumes/Second-Brain-1/AI/projects/synth-gen/scripts/synth_decryptor.py` — reads/writes/batch-decrypts .synth files

## Source Data

- **Raw .synth files:** `/Volumes/Second-Brain-1/Meta Quest/Synth/` (exactly 10,918 files)
- **Working directory:** `/Volumes/Second-Brain-1/AI/Synth/`
- **Target dataset directory:** `/Volumes/Second-Brain-1/AI/Synth/dataset/`

## IMMEDIATE ACTIONS — Execute in Order

### Step 1: Environment Setup
```bash
# Create directory structure
mkdir -p /Volumes/Second-Brain-1/AI/Synth/dataset/{raw,decrypted,extracted,specs,reports}
mkdir -p /Volumes/Second-Brain-1/AI/Synth/scripts
```

### Step 2: Inventory Source Files
```bash
# Count and list all .synth files
find /Volumes/Second-Brain-1/Meta\ Quest/Synth/ -name "*.synth" | wc -l
# Save inventory
find /Volumes/Second-Brain-1/Meta\ Quest/Synth/ -name "*.synth" > /Volumes/Second-Brain-1/AI/Synth/dataset/raw/inventory.txt
```

### Step 3: Copy (or Symlink) Files to Working Dir
```bash
# Use rsync for progress and resume capability
rsync -avh --progress /Volumes/Second-Brain-1/Meta\ Quest/Synth/*.synth /Volumes/Second-Brain-1/AI/Synth/dataset/raw/
# Verify count
ls /Volumes/Second-Brain-1/AI/Synth/dataset/raw/*.synth | wc -l
```

### Step 4: Batch Decrypt
Use the existing `synth_decryptor.py` with the password `hC2*wE5R*qQzv@a!` to decrypt all files to `dataset/decrypted/`. Log any failures.

### Step 5: Extract ZIP Contents
Each decrypted `.synth` is a ZIP. Extract each to `dataset/extracted/<uuid>/`.

### Step 6: Parse Metadata
Read `info.json` and `beatmap.meta.bin` from each extraction. Build `dataset/index.json`.

### Step 7: Inspect beatmap.data
Open a few `beatmap.data` files in hex. Document structure. Write parser.

### Step 8: Generate Report
Stats on success/failure rate, note counts, BPM distribution, difficulties.

## Constraints
- DO NOT modify originals in `/Volumes/Second-Brain-1/Meta Quest/Synth/`
- Log all failures with filenames
- Stream operations — don't load everything into RAM
- Show progress output

## Deliverables Checklist
- [ ] `dataset/raw/` has 10,918 files
- [ ] `dataset/decrypted/` has decrypted outputs
- [ ] `dataset/extracted/` has unpacked contents
- [ ] `dataset/index.json` master index exists
- [ ] `dataset/specs/beatmap.data.md` format documented
- [ ] `scripts/parse_beatmap_data.py` parser written
- [ ] `dataset/reports/ingestion-report-*.md` generated with stats
