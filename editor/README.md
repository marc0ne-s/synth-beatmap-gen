# synthgen-editor

Vite + React + TypeScript + Tailwind CSS + Tauri v2

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | Vite 6 + React 19 + TS 5.8 | RSC-ready, SWC HMR instant |
| Styling | Tailwind CSS 4 + @tailwindcss/vite | CSS-first, tree-shaken |
| State | Zustand 5 + slices | Redux-like without the boilerplate |
| 3D | R3F (@react-three/fiber) + Drei | Declarative Three.js |
| UI Primitives | shadcn/ui base (custom themed) | Radix accessibility underneath |
| Audio | wavesurfer.js | Waveform + playback |
| Hotkeys | react-hotkeys-hook | ⌘K command palette, ⌘Z undo |
| Virtualization | @tanstack/react-virtual | 100K note scroll performance |
| Tauri | v2 with HTTP, FS, Dialog, Clipboard APIs | Native file dialogs, drag-drop |
| Motion | Framer Motion | Timeline smoothness |

## Project Structure

```
editor/
├── src-tauri/                 # Rust backend
│   ├── src/
│   │   ├── main.rs            # Entry — plugin registration
│   │   ├── lib.rs             # App state, config
│   │   ├── commands/          # IPC handlers
│   │   │   ├── fs.rs          # File read/write
│   │   │   ├── project.rs     # .synthgen bundle CRUD
│   │   │   ├── audio.rs       # Audio file analysis delegation
│   │   │   └── ai.rs          # HTTP bridge to FastAPI
│   │   └── error.rs           # AppError type
│   ├── Cargo.toml             # Rust deps (tauri v2)
│   ├── tauri.conf.json        # Permissions, window config
│   └── capabilities/*.json    # Capability files (macOS, Windows)
│
├── src/
│   ├── main.tsx               # Entry — StrictMode, R3F Canvas
│   ├── App.tsx                # Router + theme provider
│   ├── index.css              # Tailwind base + CSS variables (dark)
│   │
│   ├── screens/
│   │   ├── EditorScreen.tsx   # Main layout: sidebar + timeline + 3D
│   │   ├── WelcomeScreen.tsx  # Song drop, Spotify connect, recent
│   │   ├── SettingsScreen.tsx # Model endpoint, hotkeys, appearance
│   │   └── ExportScreen.tsx   # Validation lint, platform targeting
│   │
│   ├── components/
│   │   ├── timeline/
│   │   │   ├── Timeline.tsx             # Main horizontal scroll
│   │   │   ├── WaveformTrack.tsx        # Audio waveform layer
│   │   │   ├── NoteTrack.tsx            # Note events layer
│   │   │   ├── RailTrack.tsx            # Rail events layer
│   │   │   ├── WallTrack.tsx            # Wall events layer
│   │   │   ├── Playhead.tsx             # Red line, play/pause
│   │   │   ├── Ruler.tsx                # Beat/measure markers
│   │   │   └── SnapGrid.tsx             # 1/4, 1/8, 1/16 magnets
│   │   ├── preview/
│   │   │   ├── Preview3D.tsx            # R3F Canvas wrapper
│   │   │   ├── NoteField.tsx            # VR grid floor + skybox
│   │   │   ├── NoteMesh.tsx             # Individual note glowing orb
│   │   │   ├── RailMesh.tsx             # Catmull-Rom spline tube
│   │   │   ├── WallMesh.tsx             # Block geometry
│   │   │   ├── PlayerCamera.tsx         # Orbit / first-person toggle
│   │   │   └── ParticleFX.tsx           # Hit sparkles, miss alerts
│   │   ├── inspector/
│   │   │   ├── InspectorPanel.tsx       # Side panel container
│   │   │   ├── NoteInspector.tsx        # Time/X/Y/Z/Type editors
│   │   │   ├── RailInspector.tsx        # Spline handle controls
│   │   │   └── DifficultyProfile.tsx    # Density, speed, complexity
│   │   ├── ai/
│   │   │   ├── AIGenerateButton.tsx     # One-click generate
│   │   │   ├── AIPalette.tsx            # Strength, style, difficulty
│   │   │   ├── StreamingProgress.tsx    # Real-time progress bar
│   │   │   └── SuggestionCard.tsx       # "Generate 8 bars from here"
│   │   ├── command/
│   │   │   └── CommandPalette.tsx       # ⌘K / Ctrl+K shadcn search
│   │   └── ui/                          # shadcn-style primitives
│   │       ├── glass-card.tsx           # Frosted glass container
│   │       ├── glow-text.tsx            # Neon text shadow
│   │       ├── ring-gauge.tsx           # SVG progress ring
│   │       ├── sparkline.tsx            # Mini chart
│   │       └── data-table.tsx           # Song list
│   │
│   ├── hooks/
│   │   ├── useAudioEngine.ts            # wavesurfer.js lifecycle
│   │   ├── usePlayhead.ts               # Playback transport (play/pause/scrub)
│   │   ├── useSelection.ts              # Multi-select + keyboard modifiers
│   │   ├── useSnap.ts                   # Snap to grid logic
│   │   ├── useUndo.ts                   # Undo/redo stack with branches
│   │   ├── useProject.ts                # .synthgen bundle R/W
│   │   ├── useAI.ts                     # Streaming fetch to FastAPI
│   │   └── useThreeControls.ts          # Orbit / FPS camera state
│   │
│   ├── stores/
│   │   ├── projectSlice.ts              # Current project data
│   │   ├── editorSlice.ts               # Timeline zoom, visible range
│   │   ├── playbackSlice.ts             # Playhead position, BPM
│   │   ├── selectionSlice.ts            # Selected note IDs
│   │   ├── settingsSlice.ts             # Endpoint URL, hotkey config
│   │   └── aiSlice.ts                   # Generation state, partial results
│   │
│   ├── lib/
│   │   ├── synthgen.ts                  # .synthgen read/write parser
│   │   ├── synth.ts                     # .synth export format
│   │   ├── audio.ts                     # Beat detection, BPM estimator
│   │   ├── math.ts                      # Snap, lerp, ease
│   │   ├── validation.ts              # Lint rules (overlap, bounds)
│   │   ├── difficulty.ts               # Auto-tag based on density
│   │   └── platform.ts                  # Quest limits vs PCVR limits
│   │
│   └── types/
│       ├── project.ts                   # Note, Rail, Wall interfaces
│       ├── audio.ts                     # Audio metadata, onset
│       ├── ai.ts                        # Generation request/response
│       └── api.ts                       # Tauri command types
│
├── public/
│   └── fonts/                           # JetBrains Mono variable
│
├── vite.config.ts         # Vite + @vitejs/plugin-react-swc
├── tailwind.config.ts     # Dark glassmorphism theme tokens
├── tsconfig.json
└── README.md
```

## Key Dependencies

```json
{
  "react": "^19.0.0",
  "react-dom": "^19.0.0",
  "react-router-dom": "^7.5.0",
  "three": "^0.174.0",
  "@react-three/fiber": "^9.1.0",
  "@react-three/drei": "^10.0.0",
  "framer-motion": "^12.12.0",
  "zustand": "^5.0.4",
  "wavesurfer.js": "^7.9.0",
  "cmdk": "^1.1.1",
  "react-hotkeys-hook": "^5.0.1",
  "@tanstack/react-virtual": "^3.13.6"
}
```

## Tauri Permissions (tauri.conf.json)

```json
{
  "permissions": [
    "fs:default",
    "fs:allow-read-file",
    "fs:allow-write-file",
    "fs:allow-read-dir",
    "fs:allow-copy-file",
    "dialog:default",
    "dialog:allow-open",
    "dialog:allow-save",
    "http:default",
    "clipboard-manager:default",
    "os:default",
    "path:default"
  ]
}
```