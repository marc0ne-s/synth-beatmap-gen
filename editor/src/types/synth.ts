// Type definitions for the SynthRiders .synth format
// Based on dataset/specs/beatmap.data.md and synthriders-corpus-parser skill

/** SynthRiders physical constants — must match the game exactly */
export const Constants = {
  TIME_SCALE: 20.0,
  INDEX_SCALE: 64,
  GRID_SCALE: 0.1365,
  X_OFFSET: 0.002,
  Y_OFFSET: 0.0012,
} as const

export type Difficulty = 'Easy' | 'Normal' | 'Hard' | 'Expert' | 'Master' | 'Custom'

export const Difficulties: Difficulty[] = ['Easy', 'Normal', 'Hard', 'Expert', 'Master', 'Custom']

export interface Position {
  x: number
  y: number
  z: number
}

/** A segment in a rail (sustain) — dict or raw array shape */
export type Segment =
  | { Position: [number, number, number] }
  | [number, number, number]

export interface RawNote {
  Id: string
  ComboId: number
  Position: [number, number, number]
  Segments: Segment[] | null
  Type: number | string  // 0=right, 1=left
  Direction: number  // 0=omni, 1-8=arrow directions
}

export interface BookmarksEntry {
  name: string
  time: number  // tick
}

/** Raw beatmap.meta.bin structure */
export interface RawBeatmap {
  Name: string
  Author: string
  Beatmapper: string
  BPM: number
  Offset: number
  AudioName: string
  Artwork: string
  ArtworkBytes?: string  // base64 PNG/JPG
  AudioFrecuency: number
  AudioChannels: number
  Track: Record<Difficulty, Record<string, RawNote[]>>
  Effects: Record<Difficulty, Record<string, RawNote[]>>
  Slides: Record<Difficulty, any[]>
  Lights: Record<Difficulty, number[]>
  Bookmarks: { BookmarksList: BookmarksEntry[] }
  Jumps?: Record<Difficulty, any[]>
  Crouchs?: Record<Difficulty, any[]>
  DrumSamples: any
  FilePath: string
  IsAdminOnly: boolean
  EditorVersion: string
  CustomDifficultyName: string
  CustomDifficultySpeed: number
  UsingBeatMeasure: boolean
  UpdatedWithMovementPositions: boolean
  ProductionMode: boolean
  Tags: string[]
  BeatConverted: boolean
  ModifiedTime: number
}

/** Normalised note for the editor */
export interface EditorNote {
  /** tick (INDEX_SCALE=64 per beat) */
  tick: number
  beat: number
  seconds: number
  /** synth note type: 0=right/blue, 1=left/red */
  type: number
  position: Position
  /** If true, this note is part of a rail (has segments) */
  isRail: boolean
  /** Rail path segments */
  segments: Position[]
  direction: number
  comboId: number
  id: string
}

export interface DifficultyData {
  notes: EditorNote[]
  rails: Record<string, EditorNote[]>
  slides: any[]
  lights: number[]
  effects: any[]
}

/** Editor's in-memory project */
export interface SynthProject {
  /** raw beatmap data (preserved for roundtrip — never drop unknown fields) */
  raw: RawBeatmap
  /** per-difficulty parsed data */
  difficulties: Partial<Record<Difficulty, DifficultyData>>
  /** currently active difficulty */
  activeDifficulty: Difficulty
  /** original file (for saving back) */
  sourceFile?: File
  /** is the file encrypted (Quest) */
  isEncrypted: boolean
}

// ── Conversion helpers ──────────────────────────────────────────────

export function beatToTick(beat: number): number {
  return Math.round(beat * Constants.INDEX_SCALE)
}

export function tickToBeat(tick: number): number {
  return tick / Constants.INDEX_SCALE
}

export function beatToSecond(beat: number, bpm: number): number {
  return beat * 60.0 / bpm
}

export function tickToSecond(tick: number, bpm: number): number {
  return beatToSecond(tickToBeat(tick), bpm)
}

export function secondToTick(second: number, bpm: number): number {
  return beatToTick(second * bpm / 60.0)
}

export function tickToZ(tick: number, bpm: number): number {
  return tickToSecond(tick, bpm) * Constants.TIME_SCALE
}

/** Parse segments — handles both dict-with-Position and raw-array shapes */
export function parseSegments(rawSegments: Segment[] | null): Position[] | null {
  if (!rawSegments) return null
  const result: Position[] = []
  for (const seg of rawSegments) {
    if (Array.isArray(seg)) {
      result.push({ x: seg[0], y: seg[1], z: seg[2] })
    } else if ('Position' in seg) {
      result.push({ x: seg.Position[0], y: seg.Position[1], z: seg.Position[2] })
    }
  }
  return result
}

/** Convert a difficulty from raw JSON to editor notes */
export function parseDifficulty(
  rawTrack: Record<string, RawNote[]>,
  bpm: number
): DifficultyData {
  const notes: EditorNote[] = []
  const railGroups: Record<string, EditorNote[]> = {}

  for (const [tickStr, tickNotes] of Object.entries(rawTrack)) {
    const tick = parseInt(tickStr, 10)
    const beat = tickToBeat(tick)
    const seconds = tickToSecond(tick, bpm)

    for (const raw of tickNotes) {
      const segs = parseSegments(raw.Segments)
      const note: EditorNote = {
        tick,
        beat,
        seconds,
        type: typeof raw.Type === 'string' ? parseInt(raw.Type, 10) : raw.Type,
        position: {
          x: raw.Position[0],
          y: raw.Position[1],
          z: raw.Position[2],
        },
        isRail: segs !== null,
        segments: segs || [],
        direction: raw.Direction,
        comboId: raw.ComboId,
        id: raw.Id,
      }
      notes.push(note)

      if (segs) {
        // Group rails by comboId for later manipulation
        const key = raw.ComboId.toString()
        if (!railGroups[key]) railGroups[key] = []
        railGroups[key].push(note)
      }
    }
  }

  notes.sort((a, b) => a.tick - b.tick || a.type - b.type)

  return {
    notes,
    rails: railGroups,
    slides: [],
    lights: [],
    effects: [],
  }
}
