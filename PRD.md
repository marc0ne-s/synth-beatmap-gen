# SynthGen — AI-Powered SynthRiders Beatmap Editor

## Product Requirements Document

**Version:** 0.2.0  
**Date:** 2025-04-29  
**Author:** HerBB (for Marcus)  
**Status:** Draft — revised for realistic delivery

---

## 1. Overview

SynthGen is a cross-platform desktop application that generates playable `.synth` beatmap files for SynthRiders from audio files, with an editor for manual refinement. It runs locally on Apple Silicon Macs first, then Windows, then Linux. It is free and open-source.

**The problem:** The official SynthRiders map editor is Windows-only (via Steam). Mac users cannot create maps. Generating maps manually is tedious and takes 2–6 hours per song.

**The solution:** Drop an audio file into SynthGen. The AI analyses the music and generates a complete beatmap in five difficulties. Open the editor to preview in 3D, tweak individual notes, add rails, and export a finished `.synth` file ready to play in-game.

**Key constraint:** Users provide their own audio files (MP3/WAV/OGG/FLAC). No copyrighted audio is fetched or distributed. The tool is open-source; users bear responsibility for their maps.

---

## 2. Objectives

### P0 (Must-have for v1.0)

| Objective | How We Measure |
|-----------|----------------|
| Generate a playable `.synth` file from any audio file | Time from drop to export < 150s |
| Editor with note add/edit/delete, 3D preview, timeline, undo/redo | Feature checklist |
| Export `.synth` that loads in-game without errors | Validator pass rate > 95% |
| Cross-platform: macOS first, then Windows | Successful builds on target platforms |
| Free and open-source (GPL-3.0) | License file in repo |

### P1 (v1.x, post-launch)

| Objective | When |
|-----------|------|
| Rail generation and editing | v1.1 |
| Wall generation and editing | v1.2 |
| Project save/load (`.synthgen` drafts) | v1.1 |
| Difficulty tiers fully balanced | v1.1 (Easy is known broken in v1.0) |
| Windows/Linux stable builds | v1.1 |
| AI palette (regenerate section, density slider) | v1.2 |

### P2 (Research, no commitment)

| Objective | Blocker |
|-----------|---------|
| Streaming service integration | Legal: no raw audio from Spotify/Apple Music |
| Style transfer per mapper | Need more training data |
| Model quantisation to < 200MB | Unknown effort; MPS int8 support is limited |

---

## 3. Target Users

**Primary: Budding mappers (80%)**
- Want to make maps for their favourite songs
- Don't have a Windows PC with Steam
- Willing to spend 15–30 minutes refining an AI draft
- Expect familiar UI (official editor layout where possible)

**Secondary: Pro mappers (15%)**
- Use the tool as a starting point
- Need deep manual control
- May rebuild 50% of the AI output
- Need rock-solid export

**Tertiary: Casual users (5%)**
- Just want to play their music
- Minimal editing, generate and export

---

## 4. Core User Journey (MVP)

```
User drops audio file → AI analyses (60–120s) →
→ 5 difficulties generated → Editor opens →
→ User previews in 3D, tweaks notes →
→ ⌘E exports .synth →
→ User loads .synth into SynthRiders and plays
```

**What's OUT of MVP journey:**
- Saving project drafts (⌘S is post-MVP)
- Rails/walls editing
- Streaming service integration
- Command palette
- AI palette (regenerate section)

---

## 5. Feature Specification

### 5.1 Welcome Screen

- **Drag-and-drop audio file** onto app window
- **Browse Files** button (Tauri native file dialog)
- **Recent Projects** list (persisted to `~/.synthgen/recents.json`)
- Supported: MP3, WAV, OGG, FLAC
- **No Spotify/Apple Music integration** (deferred to v2)

### 5.2 Generation Phase

- **Audio analysis** (librosa): Mel spectrogram, onset detection, BPM estimation (~10–30s)
- **Feature extraction** (`.npz` from audio) → Model inference (~30–60s)
- **Post-processing:** Difficulty-based density scaling, Temporal NMS (100ms)
- **Output:** Note arrays per difficulty
- **Progress:** Two-stage progress ("Analysing audio..." → "Generating beatmap...")
- **Cancel:** Returns to welcome screen

**Important:** The model reads pre-extracted `.npz` feature files, not raw audio. The extraction step runs before inference and adds 10–30 seconds. Total generation time for a 3-minute song is 60–120s on M4 Pro.

### 5.3 Editor — 3D Preview

- **Three.js preview** of the play area
- Notes as orbs (left=blue, right=red)
- **Rails** — rendered ONLY for v1.1+ (MVP shows notes only)
- **Walls** — rendered ONLY for v1.2+ (MVP shows notes only)
- Camera follows playhead (auto-scroll)
- Spacebar play/pause, Esc to stop
- Visual parity with in-game view (not a debug visualisation)

### 5.4 Editor — Timeline

- Horizontal scrollable timeline with waveform (wavesurfer.js)
- Playhead scrubbing
- Zoom with mouse wheel or ⌘+scroll
- Snap grid: off, 1/4, 1/8, 1/16 (cycle with G key)
- Note lanes: left hand, right hand
- Selected notes highlighted

### 5.5 Editor — Tools (MVP)

| Tool | Shortcut | Behaviour |
|------|----------|-----------|
| Select | V | Click to select, drag to multi-select, move selected notes |
| Draw | B | Click to place a note at time/position |
| Eraser | E | Click to delete note |

**Rail Draw (R) and Wall Draw (W):** v1.1+ only.

### 5.6 Inspector Panel (Right Sidebar) — MVP

- Selected note properties: time, x, y, type, hand
- Editable fields with live update
- Multi-select: bulk edit
- **No rail/wall inspector** (v1.1+)

### 5.7 Save / Load / Export

**MVP export only:**
- **⌘E** — Export `.synth` file (notes only, no rails/walls)
- Export options: difficulty selection

**v1.1+:**
- **⌘S** — Save project draft (`.synthgen`)
- **⌘O** — Open project
- **⌘⇧S** — Save As
- **Undo/redo persisted** in project file

**v1.2+:**
- Export with rails and walls

### 5.8 Undo / Redo — MVP

- Global undo/redo (⌘Z / ⌘⇧Z)
- Session-only (not persisted until v1.1)
- Minimum 50 undo steps

### 5.9 AI Palette — v1.2+

- Regenerate section (select time range, AI regenerates)
- Difficulty slider (real-time density adjustment)
- Style presets ("Flowy", "Technical") — v1.3+
- Apply / Cancel (non-destructive preview)

### 5.10 Command Palette — v1.1+

- Quick access (⌘K)
- Fuzzy search
- Keyboard-first workflow

---

## 6. Technical Architecture

### 6.1 Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + TypeScript |
| State | Zustand |
| Styling | Tailwind CSS v4 |
| 3D | Three.js + @react-three/fiber |
| Animations | Framer Motion |
| Audio viz | wavesurfer.js |
| Desktop | Tauri v2 |
| ML | PyTorch (MPS) + librosa |

### 6.2 Python Interop Strategy

**Chosen approach: HTTP localhost server** (not stdin/stdout)

Tauri spawns `python3 backend/api.py` on app startup. Frontend calls `localhost:8765` via Tauri's HTTP plugin.

**Why:** Simpler than stdin/stdout protocol, easier to debug, standard HTTP tools available. Port 8765 is unlikely to conflict.

**Risk:** If Python crashes, the frontend shows a "Generation unavailable" error. Auto-restart on crash (max 3 retries). If all retries fail, prompt user to report.

**Why not bundle Python:** Minimum viable approach. Separate Python install required (documented in README). If 2GB PyTorch bundling becomes viable later, revisit.

### 6.3 Data Flow

```
User drops audio
  → Tauri reads file path
  → Frontend calls generate_map() Tauri command
  → Rust: check Python server running (spawn if not)
  → Rust: POST /generate to localhost:8765
  → Python: extract mel features → run transformer → apply NMS
  → Python returns JSON
  → Rust returns JSON to frontend
  → Frontend: populate Zustand project store
  → Editor renders timeline + 3D preview
```

### 6.4 File Formats

**Internal project format (`.synthgen`) — v1.1+:**
```jsonc
{
  "version": "0.1.0",
  "audio_path": "/Users/marcus/Music/song.mp3",
  "bpm": 128,
  "duration": 234.5,
  "difficulties": {
    "Easy": { "notes": [...] },
    "Normal": { "notes": [...] },
    "Hard": { "notes": [...] },
    "Expert": { "notes": [...] },
    "Master": { "notes": [...] }
  }
}
```

**Export format (`.synth`) — MVP:**
Standard SynthRiders ZIP:
- `beatmap.meta.bin` — JSON with track data, notes
- `audio.ogg` — audio file
- `cover.jpg` — album art

**Note:** Current generator outputs raw JSON with `{time, type, x, y}`. Conversion to `.synth` requires seconds→ticks, z computation, and ZIP packaging. This layer (`backend/api.py`) is written but **not yet tested** with the actual game. Critical path item.

---

## 7. UI/UX Design Principles

1. **Familiar for existing mappers** — Mirror official editor where possible (V/B/E tools, space play, scroll zoom)
2. **Light, warm, organic** — Marcus's aesthetic. Soft gradients, paper texture, not dark/robotic
3. **Fast feedback** — Every action has immediate visual feedback
4. **Keyboard-first** — Every editor function has a shortcut

---

## 8. Realistic Milestones

### Phase 0: Validate End-to-End (Weeks 1–3)
- [ ] Generate one `.synth` from audio, load in-game, verify it plays
- [ ] Debug `.synth` converter (ticks, z, packaging)
- [ ] If it fails: iterate converter, try again. This is the critical gate.

### Phase 1: MVP Shell (Weeks 3–6)
- [ ] Tauri init with `src-tauri/`, file dialog, Python interop
- [ ] Welcome screen (drop audio, recent list)
- [ ] Generation progress UI (two-stage progress)
- [ ] Rust ↔ Python HTTP bridge

### Phase 2: Editor Core (Weeks 6–12)
- [ ] Timeline with waveform
- [ ] Note placement (draw/erase/select)
- [ ] 3D preview (Three.js orbs for notes)
- [ ] Playhead + playback (spacebar)
- [ ] Inspector panel (note properties)
- [ ] Export `.synth` packaging

### Phase 3: Editor Polish (Weeks 12–16)
- [ ] Undo/redo (session-only)
- [ ] Zoom and snap grid
- [ ] Multi-select and bulk edit
- [ ] Keyboard shortcuts
- [ ] Error handling (Python crash, generation failure)

### Phase 4: v1.0 Release (Weeks 16–18)
- [ ] Code signing and notarisation (macOS)
- [ ] .dmg packaging
- [ ] README, quickstart guide
- [ ] Open-source on GitHub
- [ ] Manual QA on clean Mac

### Phase 5: Rails + Walls (Weeks 18–24)
- [ ] Rail generation heuristic
- [ ] Rail editing (draw segments)
- [ ] Wall editing
- [ ] Export with rails/walls

### Phase 6: Project Save + AI Palette (Weeks 24–30)
- [ ] `.synthgen` project save/load
- [ ] Undo/redo persistence
- [ ] AI palette (regenerate section)
- [ ] Difficulty slider

### Phase 7: Cross-Platform (Weeks 30–36)
- [ ] Windows build
- [ ] Linux build
- [ ] Installer packaging (.msi, .deb)

**Total: ~8–9 months for v1.0. Rails/walls in months 4–6. Cross-platform in months 7–9.**

---

## 9. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `.synth` converter doesn't work with game | High | **High** | **Phase 0 is strictly for this.** If it fails, nothing else matters. |
| Tauri + Python interop is flaky | Medium | Medium | HTTP localhost; auto-restart; clear error messages |
| AI quality insufficient | High | Medium | MVP is v1.0, not perfect. Pro mapper feedback loop in v1.x |
| Easy difficulty broken | Medium | **High** | Known issue. Fix in v1.1. v1.0 ships with disclaimer |
| No code signing for Mac | High | Medium | $99/year Apple Developer Program. Required for distribution |
| Model weight size (500MB) | Low | Low | User downloads separately, or we use quantisation later |
| Rails/walls don't generate | Medium | **High** | Heuristic-only for v1.1. Full model in v1.3+ |

---

## 10. Success Metrics (v1.0)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Generation time (3-min song) | < 150s | Instrumented |
| Playable export rate | > 95% | Automated validator |
| Easy pass rate | > 50% | Feasibility checker (known broken, 37% currently) |
| Editor crash rate | < 5% | Error reporting (MVP acceptable) |
| GitHub stars (month 1) | > 50 | Social proof |

**Pro mapper blind test v1.2+:** Target >60% "human" rating (not in v1.0).

---

## 11. Dependencies

| Dependency | Version | Source | Notes |
|------------|---------|--------|-------|
| React | 19.0+ | npm | |
| Three.js | 0.174+ | npm | |
| Tauri | 2.5+ | Cargo | |
| PyTorch | 2.0+ | pip | MPS on Apple Silicon |
| librosa | 0.10+ | pip | Audio feature extraction |
| numpy | 1.24+ | pip | |

**User prerequisites for v1.0:**
- macOS 13.0+ (MPS requires recent macOS)
- Python 3.11+ with PyTorch, librosa installed
- Apple Developer account for code signing ($99/year)

---

## 12. Appendices

### A. Known Issues (Do not fix for v1.0)

- **Easy difficulty left-bias:** 78/22 left-hand split. Fix in v1.1.
- **Rails/walls not generated:** Model outputs notes only. Fix in v1.1+.
- **Model quantisation:** 500MB weights. Explore in v1.3+.

### B. Asset Locations

- Model checkpoints: `models/checkpoints/transformer_phase12b_ep5.pt`
- Generator: `scripts/generate_gold_standard.py`
- Converter (API): `backend/api.py` (new, untested)
- Feasibility checker: `scripts/feasibility_checker.py`
- Editor scaffolding: `editor/` (React + Tauri, missing components)

### C. What's Built vs What's Missing

**Built:**
- [x] Training pipeline and working model (Phase 12b)
- [x] Generator script (outputs JSON)
- [x] Feasibility checker
- [x] Editor scaffolding (WelcomeScreen, EditorScreen skeleton)
- [x] Python API server (new)

**Missing (critical path):**
- [ ] `.synth` converter tested with actual game
- [ ] Tauri `src-tauri/` compiled
- [ ] Timeline component
- [ ] 3D Preview component (Three.js orbs)
- [ ] InspectorPanel component
- [ ] Undo/redo system
- [ ] Audio playback
- [ ] File save/load
- [ ] Export packaging

### D. Open Questions

1. **How do we verify the `.synth` converter works?** Need to test in SynthRiders ASAP.
2. **Can we train a rail/wall model from existing maps?** Need `.synth` corpus with rails/walls.
3. **Should we distribute on Homebrew Cask?** Easier than .dmg download.
4. **What's the minimum viable Python install script?** `pip install torch librosa numpy fastapi uvicorn` — is that enough?

---

**End of PRD v0.2.0.**
