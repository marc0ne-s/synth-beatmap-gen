# SynthRiders: Quick Decrypt Test (Validate Pipeline)
## Run FIRST before full corpus ingestion

**Goal:** Decrypt ~5 .synth files, validate structure, inspect `beatmap.data` format.

## Password
`hC2*wE5R*qQzv@a!`

## Scripts
- `/Volumes/Second-Brain-1/AI/projects/synth-gen/scripts/synth_decryptor.py` — tested_decryptor

## Step 1: Sample Selection
```bash
mkdir -p /Volumes/Second-Brain-1/AI/Synth/test-decrypt/
find /Volumes/Second-Brain-1/Meta\ Quest/Synth/ -name "*.synth" | head -5 | xargs -I {} cp {} /Volumes/Second-Brain-1/AI/Synth/test-decrypt/
ls -lh /Volumes/Second-Brain-1/AI/Synth/test-decrypt/
```

## Step 2: Decrypt Sample
```bash
cd /Volumes/Second-Brain-1/AI/Synth/
python3 /Volumes/Second-Brain-1/AI/projects/synth-gen/scripts/synth_decryptor.py --input test-decrypt/ --output test-decrypt/out/ --password "hC2*wE5R*qQzv@a!"
# OR if decryptor lacks CLI args, write a 5-line Python wrapper:
ls test-decrypt/*.synth | head -5
```

## Step 3: Inspect Output
- Are the outputs valid ZIP files? Check with `file` command
- Can you unzip them? `unzip -l <file>` to list contents without extracting
- What files are inside? Look for: `beatmap.meta.bin`, `beatmap.data`, `cover.jpg`, `info.json`

## Step 4: Extract One Sample
```bash
mkdir -p test-decrypt/extracted/sample1/
cd test-decrypt/extracted/sample1/
# Replace with actual decrypted ZIP path
unzip </path/to/decrypted/file>
ls -lah
```

## Step 5: Inspect Key Files
1. **info.json** — cat it. Extract: title, artist, BPM, difficulty
2. **beatmap.meta.bin** — check file size, run `xxd | head -20`, document anything readable
3. **beatmap.data** — this is the binary note data. Run:
   ```bash
   xxd beatmap.data | head -50
   file beatmap.data
   ls -lh beatmap.data
   ```
   Look for patterns: repeating structures, timestamps, lane/position values.

## Step 6: Validate
- [ ] Decryptor works on real files (not just test cases)
- [ ] Output is valid ZIP
- [ ] Expected files are present inside
- [ ] `beatmap.data` is binary and non-empty
- [ ] `info.json` is readable JSON with song metadata

## Step 7: Report
Post a concise summary:
- File sizes (raw .synth vs decrypted ZIP)
- Contents of info.json (one example)
- First 50 lines of `xxd` on beatmap.data
- Whether the pipeline looks viable for 10,918 files

**Only proceed to full ingestion if this test passes cleanly.**
