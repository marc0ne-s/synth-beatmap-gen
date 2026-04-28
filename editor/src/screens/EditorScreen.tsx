import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { useHotkeys } from 'react-hotkeys-hook'
import { motion, AnimatePresence } from 'framer-motion'
import { useProject } from '../stores/project'
import { useEditor } from '../stores/editor'
import Timeline from '../components/timeline/Timeline'
import Playhead from '../components/timeline/Playhead'
import Preview3D from '../components/preview/Preview3D'
import InspectorPanel from '../components/inspector/InspectorPanel'
import CommandPalette from '../components/command/CommandPalette'
import AIPalette from '../components/ai/AIPalette'

export default function EditorScreen() {
  const { project } = useProject()
  const { activeScreen, show3D, togglePlay, stop, setTool, activeTool, setSnap, snapGrid } = useEditor()

  useHotkeys('space', (e) => { e.preventDefault(); togglePlay() }, [togglePlay])
  useHotkeys('esc', stop, [stop])
  useHotkeys('v', () =>
003e setTool('select'), [setTool])
  useHotkeys('b', () =>
003e setTool('draw'), [setTool])
  useHotkeys('e', () =>
003e setTool('eraser'), [setTool])
  useHotkeys('g', () =>
003e setSnap(snapGrid === '1/8' ? '1/16' : snapGrid === '1/16' ? '1/4' : '1/8'), [snapGrid, setSnap])

  if (activeScreen !== 'editor') return null

  return (
    <div className="h-screen flex flex-col overflow-hidden"
      style={{ background: 'var(--color-bg)' }}
    >
      <CommandPalette />

      {/* Toolbar */}
      <header className="flex items-center gap-4 px-5 py-3 border-b"
        style={{ borderColor: 'rgba(100,160,255,0.06)' }}
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
            style={{ background: 'linear-gradient(135deg, #e040fb, #42a5f5)', color: 'white' }}
          >◈</div>
          <div>
            <div className="text-sm font-semibold leading-tight">{project.title}</div>
            <div className="text-xs"
              style={{ color: 'var(--color-text-dim)' }}
            >{project.artist} · {project.bpm} BPM</div>
          </div>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <ToolButton active={activeTool === 'select'} onClick={() =>
003e setTool('select')} label="Select" shortcut="V" />
          <ToolButton active={activeTool === 'draw'} onClick={() =>
003e setTool('draw')} label="Draw" shortcut="B" />
          <ToolButton active={activeTool === 'eraser'} onClick={() =>
003e setTool('eraser')} label="Erase" shortcut="E" />
          <div className="w-px h-6 mx-2"
            style={{ background: 'rgba(100,160,255,0.1)' }}
          />
          <SnapButton snap={snapGrid} onClick={() =>
003e setSnap(snapGrid === '1/8' ? '1/16' : snapGrid === '1/16' ? '1/4' : '1/8')} />
          <AIButton />
          <div className="w-px h-6 mx-2"
            style={{ background: 'rgba(100,160,255,0.1)' }}
          />
          <ExportButton />
        </div>
      </header>

      {/* Main body */}
      <div className="flex-1 flex min-h-0">
        {/* Timeline */}
        <div className="flex-1 flex flex-col min-w-0"
        >
          <div className="flex-1 flex overflow-hidden"
          >
            {show3D && (
              <div className="w-80 shrink-0 border-r"
                style={{ borderColor: 'rgba(100,160,255,0.06)' }}
              >
                <div className="h-full">
                  <Canvas
                    camera={{ position: [0, 0, 5], fov: 60 }}
                    style={{ background: 'transparent' }}
                  >
                    <Preview3D />
                  </Canvas>
                </div>
              </div>
            )}
            <div className="flex-1 flex flex-col min-w-0"
            >
              <div className="flex-1 relative overflow-hidden"
              >
                <Timeline />
                <Playhead />
              </div>
            </div>
          </div>
        </div>

        {/* Inspector */}
        <div className="w-72 shrink-0 border-l"
          style={{ borderColor: 'rgba(100,160,255,0.06)' }}
        >
          <InspectorPanel />
        </div>
      </div>

      {/* AI Palette (floating, conditional) */}
      <AIPalette />
    </div>
  )
}

// Subcomponents
function ToolButton({ active, onClick, label, shortcut }: { active: boolean; onClick: () =>
003e void; label: string; shortcut: string }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-md text-xs font-medium transition-all"
      style={{
        background: active ? 'rgba(66, 165, 245, 0.15)' : 'transparent',
        color: active ? 'var(--color-blue)' : 'var(--color-text-dim)',
        border: active ? '1px solid rgba(66, 165, 245, 0.25)' : '1px solid transparent',
      }}
    >
      {label} <span className="ml-1 opacity-50">{shortcut}</span>
    </button>
  )
}

function SnapButton({ snap, onClick }: { snap: string; onClick: () =>
003e void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-md text-xs font-medium transition-all"
      style={{
        background: 'rgba(255, 167, 38, 0.08)',
        color: 'var(--color-amber)',
        border: '1px solid rgba(255, 167, 38, 0.15)',
      }}
    >Snap {snap}</button>
  )
}

function AIButton() {
  return (
    <button
      className="px-4 py-1.5 rounded-md text-xs font-bold transition-all hover:brightness-110"
      style={{
        background: 'linear-gradient(135deg, rgba(224,64,251,0.15), rgba(66,165,245,0.15))',
        color: 'var(--color-magenta)',
        border: '1px solid rgba(224, 64, 251, 0.2)',
      }}
    >✨ Generate</button>
  )
}

function ExportButton() {
  return (
    <button
      className="px-4 py-1.5 rounded-md text-xs font-semibold transition-all hover:brightness-110"
      style={{
        background: 'rgba(0, 229, 160, 0.1)',
        color: 'var(--color-green)',
        border: '1px solid rgba(0, 229, 160, 0.2)',
      }}
    >Export</button>
  )
}