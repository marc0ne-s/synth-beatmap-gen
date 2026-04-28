# SynthGen Editor — Design Requirements

**Version:** 0.1-draft  
**Date:** 2026-04-27  
**Status:** In design — pending Phase 12 results  

---

## 1. Elevator Pitch

SynthGen Editor is a **cross-platform desktop beatmap authoring tool** for Synth Riders. It combines AI-generated beatmap suggestions with a professional human-in-the-loop interface. Open any song → get AI beatmap in seconds → refine visually → one-click export to Quest/Steam.

---

## 2. Core Philosophy

| Principle | Implementation |
|-----------|---------------|
| **AI-first, human-refined** | AI generates raw beatmaps; editor provides surgical editing tools |
| **Desktop-native** | Not a browser toy — professional tool with native performance |
| **Cross-platform** | Mac first (where Marcus works), then Windows/Linux via Electron/Tauri |
| **3D-native** | Real-time 3D preview matching in-game perspective |
| **Streaming-aware** | Pull track metadata from Spotify/Apple Music for instant indexing |

---

## 3. Feature Matrix

### 3.1 Input & Discovery

| Feature | Priority | Description |
|---------|----------|-------------|
| **Drag-drop audio** | P0 | Drop any MP3/WAV/OGG → auto-analysis → AI generate |
| **Spotify connect** | P1 | OAuth to Spotify → browse playlists → pick song → auto-populate metadata |
| **Apple Music connect** | P1 | Same for Apple Music (macOS share extension) |
| **Local library scan** | P1 | Index ~/Music → show all tracks with inferred BPM/tempo |
| **YouTube audio** | P2 | Paste URL → yt-dlp → generate (respects TOS, user-hosted) |
| **Existing .synth import** | P0 | Open existing beatmaps for remixing or study |

### 3.2 AI Generation Pipeline

| Feature | Priority | Description |
|---------|----------|-------------|
| **One-click generate** | P0 | Single button → full beatmap (notes + rails + walls) |
| **Difficulty presets** | P0 | Easy / Normal / Hard / Expert / Master / Custom |
| **Style transfer** | P1 | "Like [artist/map name]" → finetune on that style |
| **Partial generate** | P1 | Generate only notes, only walls, only rails |
| **Regenerate section** | P1 | Select bars 32-48 → AI suggest alternatives |
| **Strength slider** | P2 | Conservative → Aggressive AI risk appetite |

### 3.3 Visual Editor

| Feature | Priority | Description |
|---------|----------|-------------|
| **Timeline view** | P0 | Piano-roll-style horizontal timeline (time → x-axis) |
| **3D preview pane** | P0 | Real-time 3D view: notes fly toward player in game-space |
| **Multi-select** | P0 | Drag rectangle → select notes → delete/move/quantize |
| **Note inspector** | P0 | Click note → side panel: time, x, y, z, rail, hand, type |
| **Rail editor** | P1 | Visual rail path drawing with bezier handles |
| **Wall editor** | P1 | Click-and-drag wall regions with live preview |
| **Snap grid** | P0 | 1/4, 1/8, 1/16 beat snap with visual toggle |
| **Undo/redo** | P0 | Full history tree (branching for experiments) |
| **Zoom** | P0 | Mousewheel zoom timeline ↔, ⌘+scroll pan |
| **Multi-track** | P2 | Show notes/rails/walls on separate editable layers |

### 3.4 Audio Sync & Playback

| Feature | Priority | Description |
|---------|----------|-------------|
| **Scrubbable waveform** | P0 | Audio waveform under timeline, clickable to jump |
| **Beat markers** | P0 | Vertical beat lines with measure numbers |
| **Playhead** | P0 | Red line, play/pause/loop controls |
| **Slow-motion play** | P1 | 0.5x, 0.25x for precise editing |
| **Count-in** | P2 | 1-bar pre-roll when hitting play |

### 3.5 3D Preview Engine

| Feature | Priority | Description |
|---------|----------|-------------|
| **Real-time render** | P0 | 60fps Three.js or raw WebGL, matches game camera |
| **VR headset preview** | P2 | Quest Link / Air Link → see map in-headset while editing |
| **Camera orbit** | P0 | Orbit, pan, first-person (VR camera placement) |
| **Note material** | P0 | Match Synth Riders shaders (glow, color by note type) |
| **Rail rendering** | P1 | Smooth spline rails with particle trail |
| **Wall rendering** | P1 | Semi-transparent shader matching game |
| **Play test mode** | P1 | Click play → watch notes fly → catch/collide test |
| **Hand tracking preview** | P3 | Show estimated hand trajectories |

### 3.6 Export & Validation

| Feature | Priority | Description |
|---------|----------|-------------|
| **One-click export** | P0 | Export valid .synth file with all required fields |
| **Validation lint** | P0 | Auto-check: impossible rails, overlapping notes, out-of-bounds |
| **Difficulty auto-tag** | P1 | Estimate difficulty from note density and speed |
| **Platform targeting** | P1 | Quest 2/3/Pro, PCVR specs (note count limits, performance) |
| **Upload to platform** | P2 | Upload to Synth Riders official community hub |
| **Custom export path** | P2 | Direct to headset: AirDrop to Quest, or wired ADB push |

### 3.7 Project Management

| Feature | Priority | Description |
|---------|----------|-------------|
| **Project files** | P0 | `.synthgen` bundle: audio + beatmap + metadata + backups |
| **Version history** | P1 | Git-style branching: experiments, style variants |
| **Auto-save** | P0 | Every 30s + on every significant operation |
| **Recovery** | P1 | Crash → reopen → restore exact playhead position |
| **Collaboration** | P2 | Shared projects, comments, multiplayer editing |

---

## 4. Platform Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-PLATFORM TARGETING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐              │
│  │  macOS   │  →     │  Tauri   │  →     │  Windows │              │
│  │  (M1/M2) │        │  Core    │        │  (x86)   │              │
│  │          │        │  (Rust)  │        │          │              │
│  │  ┌────┐  │        │          │        │  ┌────┐  │              │
│  │  │ Web │ │        │  ┌────┐  │        │  │Web │ │              │
│  │  │View│  │        │  │Rust│  │        │  │View│  │              │
│  │  └────┘  │        │  │API │  │        │  └────┘  │              │
│  │     ↕    │        │  └────┘  │        │     ↕    │              │
│  │  3D:     │        │     ↕    │        │  3D:     │              │
│  │  Three.js│        │  Native  │        │  Three.js│              │
│  │  (WebGL) │        │  OS APIs │        │  (WebGL) │              │
│  └──────────┘        └──────────┘        └──────────┘              │
│     PRIORITY 1          SHARED CORE          PRIORITY 2            │
│                                                                     │
│  Electron fallback if Tauri doesn't cut it for 3D perf.          │
│  Linux: same build as Windows, community-supported.                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Tech stack:**
- **Framework:** Tauri v2 (Rust backend + Web frontend)
- **Frontend:** React + TypeScript + TailwindCSS
- **3D Engine:** Three.js (WebGL, GPU-accelerated)
- **Audio:** Web Audio API + wavesurfer.js for waveform
- **AI Bridge:** HTTP to local FastAPI server (Phase 15 deploy)
- **State:** Zustand for UI, SQLite embedded for project data
- **Storage:** OS-native file dialogs via Tauri APIs

---

## 5. UI Layout Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│  File  Edit  View  Track  AI  Export                              │  ← Menu Bar
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────────────────────────────────┐  │
│  │                │  │                                        │  │
│  │   3D PREVIEW   │  │         TIMELINE EDITOR                 │  │
│  │                │  │                                        │  │
│  │   [Notes fly   │  │  Waveform │█ ▓ ░ ░ ▓ █│              │  │
│  │    toward     │  │  ─────────┼──┼──┼──┼──┼───            │  │
│  │    camera]    │  │  Notes    │●  ●●   ●   ●●              │  │
│  │                │  │  Rails    │~~~~  ~~~  ~~~~            │  │
│  │                │  │  Walls    │ ███  ███  ██              │  │
│  │                │  │  ─────────┼──┼──┼──┼──┼───            │  │
│  │                │  │           ▲ playhead                   │  │
│  └────────────────┘  └────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────┐  ┌────────────────────────────────────────────┐  │
│  │  Track List    │  │  Properties Inspector                     │  │
│  │  ────────────  │  │  ─────────────────────                     │  │
│  │  Song: XXXX    │  │  Selected: Note #42                       │  │
│  │  BPM: 128      │  │  Time: 32.4s    X: -0.3                    │  │
│  │  Key: Em       │  │  Y: 1.2m        Z: 2.0m                   │  │
│  │  Difficulty:   │  │  Type: Left Hand / Blue Rail              │  │
│  │  Expert        │  │  Rail ID: r_7                              │  │
│  └────────────────┘  └────────────────────────────────────────────┘  │
│                                                                     │
│  [Generate with AI] [Undo] [Redo] [Play] [Export .synth]          │  ← Toolbar
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Model

### 6.1 Project File (`*.synthgen`)

```json
{
  "version": "1.0",
  "metadata": {
    "title": "Cyberpunk Dreams",
    "artist": "Neon Pulse",
    "bpm": 128.5,
    "duration": 234.5,
    "difficulty": "Expert",
    "tags": ["electronic", "fast", "rails"],
    "ai_model_version": "phase12_ep7",
    "ai_confidence": 0.78
  },
  "audio": {
    "format": "mp3",
    "hash": "sha256:abc123...",
    "file": "audio/master.mp3"
  },
  "beatmap": {
    "notes": [...],
    "rails": [...],
    "walls": [...],
    "events": [...]
  },
  "checkpoints": ["gen_1", "gen_2", "manual_edit_1"],
  "export_config": {
    "platform": "quest3",
    "synth_version": "2.5.1"
  }
}
```

### 6.2 Note Entity

```typescript
interface Note {
  id: string;
  time: number;       // seconds, float
  x: number;          // -3.0 to +3.0
  y: number;          // 0.0 to 3.0
  z: number;          // depth coordinate
  hand: 'left' | 'right' | 'either';
  type: 'standard' | 'rail_start' | 'rail_mid' | 'rail_end' | 'mine';
  color: string;      // hex
  railId?: string;    // null if standalone
  velocity?: number;    // approach speed
}
```

---

## 7. AI Integration Points

| Trigger | Flow | Latency Target |
|---------|------|--------------|
| First song load | Audio → FastAPI → AI generate → stream back notes | < 10s |
| Regenerate section | Selected bars → API → partial replace | < 3s |
| Style transfer | User picks reference map → API finetunes on embedding | < 30s |
| Real-time suggestion | User pauses → AI suggests next 8 bars | < 2s (cached) |
| Export validation | Beatmap → API → lint report | < 1s |

---

## 8. Open Questions

1. **Does Tauri's WebView handle Three.js at 60fps?** Need prototype.
2. **Should we build native Metal/Vulkan renderer instead** of WebGL for 3D preview?
3. **Spotify API:** Do we need commercial partnership for beatmap generation metadata?
4. **Apple Music:** Raw audio access is restricted — metadata-only integration?
5. **VR preview:** Is Quest Link realistic for editor workflow, or should we stream preview video?
6. **Collaboration:** Real-time multiplayer editing — WebRTC or operational transforms?

---

## 9. Success Criteria

| Milestone | Metric |
|-----------|--------|
| **Alpha** | Generate → edit → export 1 valid .synth in under 2 min |
| **Beta** | 3D preview at 60fps, timeline editing feels like Ableton |
| **1.0** | User generates map, exports to Quest, plays in Synth Riders without editing |

---

## 10. Phase Dependencies

```
Phase 12 (now) → Transformer model quality
        ↓
Phase 13 (this doc) → Editor design kickoff
        ↓
Phase 14 → Auto-playtest loop (headset in the loop)
        ↓
Phase 15 → Cloud API → editor connects to hosted model
```

Editor development can start **in parallel** with Phase 12 training using synthetic/placeholder beatmaps.

---

**Document owner:** Marcus / HerBB  
**Next review:** After Phase 12 epoch 5 checkpoint