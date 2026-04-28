import { useCallback } from 'react'
import { motion } from 'framer-motion'

interface WelcomeScreenProps {
  onOpenProject: (path: string) => void
  onNewProject: () => void
  onDropAudio: (files: FileList) => void
}

// Mock recent projects — would read from ~/.synthgen/recents.json via Tauri
const RECENTS = [
  { title: 'Neon Pulse', artist: 'Cyber Dreams', modified: '2h ago', difficulty: 'Expert' },
  { title: 'System Shock', artist: 'Glitch Mode', modified: '1d ago', difficulty: 'Master' },
]

export default function WelcomeScreen({ onOpenProject, onNewProject, onDropAudio }: WelcomeScreenProps) {
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files.length > 0) {
      onDropAudio(e.dataTransfer.files)
    }
  }, [onDropAudio])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-10 p-8"
    >
      {/* Brand */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="text-center"
      >
        <div className="inline-block w-20 h-20 rounded-2xl mb-6"
          style={{
            background: 'linear-gradient(135deg, #e040fb, #42a5f5)',
            boxShadow: '0 0 50px rgba(224,64,251,0.2), inset 0 0 30px rgba(255,255,255,0.1)',
          }}
        >
          <div className="flex items-center justify-center h-full text-3xl font-bold text-white">◈</div>
        </div>
        <h1 className="text-4xl font-bold tracking-tight"
          style={{ fontFamily: 'var(--font-mono)' }}
        >
          SYNTH<span style={{ color: 'var(--color-cyan)' }}>RIDERS</span>
        </h1>
        <p className="mt-3 text-base"
          style={{ color: 'var(--color-text-dim)' }}
        >
          AI-Powered Beatmap Generation for VR Rhythm
        </p>
      </motion.div>

      {/* Drop Zone */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.15, duration: 0.5 }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="w-full max-w-xl rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer"
        style={{
          borderColor: 'rgba(100, 160, 255, 0.15)',
          background: 'linear-gradient(145deg, rgba(19,27,44,0.5), rgba(8,12,20,0.7))',
        }}
        onDragEnter={(e) => {
          const target = e.currentTarget
          target.style.borderColor = 'rgba(0, 229, 160, 0.4)'
          target.style.background = 'linear-gradient(145deg, rgba(0,229,160,0.05), rgba(8,12,20,0.7))'
        }}
        onDragLeave={(e) => {
          const target = e.currentTarget
          target.style.borderColor = 'rgba(100, 160, 255, 0.15)'
          target.style.background = 'linear-gradient(145deg, rgba(19,27,44,0.5), rgba(8,12,20,0.7))'
        }}
      >
        <div className="text-5xl mb-4">🎵</div>
        <h3 className="text-xl font-semibold mb-2">Drop a song to generate</h3>
        <p style={{ color: 'var(--color-text-dim)' }} className="mb-6">MP3, WAV, OGG, or FLAC — AI handles the rest</p>
        <div className="flex gap-3 justify-center">
          <button
            onClick={() => {/* Tauri dialog open */}}
            className="px-5 py-2.5 rounded-lg font-medium text-sm transition-all hover:brightness-110"
            style={{
              background: 'rgba(66, 165, 245, 0.12)',
              border: '1px solid rgba(66, 165, 245, 0.2)',
              color: 'var(--color-blue)',
            }}
          >Browse Files</button>
          <button
            onClick={() => {/* Spotify OAuth */}}
            className="px-5 py-2.5 rounded-lg font-medium text-sm transition-all hover:brightness-110"
            style={{
              background: 'rgba(29, 185, 84, 0.1)',
              border: '1px solid rgba(29, 185, 84, 0.2)',
              color: '#1db954',
            }}
          >🎧 Spotify</button>
        </div>
      </motion.div>

      {/* Recent Projects */}
      {RECENTS.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="w-full max-w-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold uppercase tracking-widest"
              style={{ color: 'var(--color-text-faint)' }}
            >Recent Projects</span>
            <button
              onClick={onNewProject}
              className="text-xs font-medium transition-colors hover:brightness-120"
              style={{ color: 'var(--color-cyan)' }}
            >+ New Blank</button>
          </div>
          <div className="flex flex-col gap-2">
            {RECENTS.map((r, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 + i * 0.08 }}
                className="flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all hover:brightness-110"
                style={{
                  background: 'rgba(19, 27, 44, 0.4)',
                  border: '1px solid rgba(100, 160, 255, 0.06)',
                }}
                onClick={() => onOpenProject(r.title)}
              >
                <div className="w-10 h-10 rounded-lg flex items-center justify-center text-lg"
                  style={{ background: 'rgba(0, 229, 160, 0.08)' }}
                >🎧</div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">{r.title}</div>
                  <div className="text-xs truncate"
                    style={{ color: 'var(--color-text-dim)' }}
                  >{r.artist} · {r.modified}</div>
                </div>
                <div
                  className="text-xs px-2 py-1 rounded-md font-medium"
                  style={{
                    background: r.difficulty === 'Master' ? 'rgba(255,82,82,0.1)' : 'rgba(255,167,38,0.1)',
                    color: r.difficulty === 'Master' ? 'var(--color-red)' : 'var(--color-amber)',
                  }}
                >{r.difficulty}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Footer shortcuts */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="flex gap-6 text-xs"
        style={{ color: 'var(--color-text-faint)' }}
      >
        <span>⌘K Command Palette</span>
        <span>⌘O Open…</span>
        <span>⌘, Settings</span>
      </motion.div>
    </div>
  )
}