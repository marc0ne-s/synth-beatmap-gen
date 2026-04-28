import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

export interface Note {
  id: string
  time: number
  x: number
  y: number
  z: number
  hand: 'left' | 'right' | 'either'
  type: 'standard' | 'rail_start' | 'rail_mid' | 'rail_end' | 'mine'
  color: string
  railId?: string
  velocity?: number
}

export interface ProjectState {
  path: string | null
  title: string
  artist: string
  bpm: number
  duration: number
  difficulty: 'Easy' | 'Normal' | 'Hard' | 'Expert' | 'Master' | 'Custom'
  notes: Note[]
  dirty: boolean
}

interface ProjectStore {
  project: ProjectState
  setProject: (p: Partial<ProjectState>) => void
  addNote: (n: Note) => void
  removeNote: (id: string) => void
  updateNote: (id: string, patch: Partial<Note>) => void
  clear: () => void
}

const empty: ProjectState = {
  path: null,
  title: 'Untitled',
  artist: 'Unknown',
  bpm: 128,
  duration: 0,
  difficulty: 'Expert',
  notes: [],
  dirty: false,
}

export const useProject = create<ProjectStore>()(
  subscribeWithSelector((set) => ({
    project: empty,
    setProject: (p) => set((state) => ({ project: { ...state.project, ...p, dirty: true } })),
    addNote: (n) => set((state) => ({
      project: { ...state.project, notes: [...state.project.notes, n].sort((a, b) => a.time - b.time), dirty: true }
    })),
    removeNote: (id) => set((state) => ({
      project: { ...state.project, notes: state.project.notes.filter(n => n.id !== id), dirty: true }
    })),
    updateNote: (id, patch) => set((state) => ({
      project: {
        ...state.project,
        notes: state.project.notes.map(n => n.id === id ? { ...n, ...patch } : n),
        dirty: true,
      }
    })),
    clear: () => set({ project: empty }),
  }))
)