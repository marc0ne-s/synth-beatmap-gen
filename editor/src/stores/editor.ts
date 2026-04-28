import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

export interface EditorState {
  activeScreen: 'welcome' | 'editor' | 'settings' | 'export'
  timelineZoom: number
  visibleRange: [number, number]
  playhead: number
  isPlaying: boolean
  snapGrid: 'off' | '1/4' | '1/8' | '1/16'
  activeTool: 'select' | 'draw' | 'eraser'
  show3D: boolean
  showWaveform: boolean
}

interface EditorStore extends EditorState {
  setScreen: (s: EditorState['activeScreen']) => void
  setZoom: (z: number) => void
  panRange: (delta: number) => void
  setPlayhead: (t: number) => void
  togglePlay: () => void
  stop: () => void
  setSnap: (s: EditorState['snapGrid']) => void
  setTool: (t: EditorState['activeTool']) => void
  toggle3D: () => void
  toggleWaveform: () => void
}

export const useEditor = create<EditorStore>()(
  subscribeWithSelector((set) => ({
    activeScreen: 'welcome',
    timelineZoom: 1,
    visibleRange: [0, 60],
    playhead: 0,
    isPlaying: false,
    snapGrid: '1/8',
    activeTool: 'select',
    show3D: true,
    showWaveform: true,

    setScreen: (s) => set({ activeScreen: s }),
    setZoom: (z) => set((state) => ({ timelineZoom: Math.max(0.1, Math.min(10, z)) })),
    panRange: (delta) => set((state) => ({
      visibleRange: [
        Math.max(0, state.visibleRange[0] + delta),
        state.visibleRange[1] + delta,
      ] as [number, number],
    })),
    setPlayhead: (t) => set({ playhead: t }),
    togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
    stop: () => set({ isPlaying: false, playhead: 0 }),
    setSnap: (s) => set({ snapGrid: s }),
    setTool: (t) => set({ activeTool: t }),
    toggle3D: () => set((state) => ({ show3D: !state.show3D })),
    toggleWaveform: () => set((state) => ({ showWaveform: !state.showWaveform })),
  }))
)