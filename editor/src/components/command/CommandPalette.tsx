// CommandPalette — ⌘K command palette (stub, cmdk-ready)
import { useState, useEffect, useCallback } from 'react'
import { Command as CmdkCommand } from 'cmdk'
import { motion, AnimatePresence } from 'framer-motion'
import { useEditor } from '../../stores/editor'

// Commands available in the palette
const commands = [
  { id: 'load', label: 'Load .synth File', shortcut: '⌘O', action: () => {/* file picker */} },
  { id: 'export', label: 'Export .synth', shortcut: '⌘S', action: () => {} },
  { id: 'play', label: 'Play / Pause', shortcut: 'Space', action: () => useEditor.getState().togglePlay() },
  { id: 'editor', label: 'Go to Editor', shortcut: '⌘1', action: () => useEditor.getState().setScreen('editor') },
  { id: 'welcome', label: 'Back to Home', shortcut: '⌘2', action: () => useEditor.getState().setScreen('welcome') },
  { id: 'toggle3d', label: 'Toggle 3D View', shortcut: '⌘3', action: () => useEditor.getState().toggle3D() },
  { id: 'togglewave', label: 'Toggle Waveform', shortcut: '⌘4', action: () => useEditor.getState().toggleWaveform() },
  { id: 'snap', label: 'Cycle Snap Grid', shortcut: 'G', action: () => {
    const s = useEditor.getState()
    s.setSnap(s.snapGrid === '1/8' ? '1/16' : s.snapGrid === '1/16' ? '1/4' : '1/8')
  }},
]

export default function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault()
      setOpen((o) => !o)
    }
    if (e.key === 'Escape') setOpen(false)
  }, [])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const filtered = commands.filter((c) =>
    c.label.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]"
          style={{ background: 'rgba(0,0,0,0.6)' }}
          onClick={() => setOpen(false)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -10 }}
            transition={{ duration: 0.15 }}
            className="w-full max-w-lg rounded-xl overflow-hidden"
            style={{
              background: 'var(--color-bg-card)',
              border: '1px solid var(--color-border-glow)',
              boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <CmdkCommand>
              <div style={{ borderBottom: '1px solid var(--color-border)' }}>
                <CmdkCommand.Input
                  autoFocus
                  value={query}
                  onValueChange={setQuery}
                  placeholder="Type a command…"
                  className="w-full px-4 py-3 text-sm bg-transparent outline-none"
                  style={{ color: 'var(--color-text)' }}
                />
              </div>
              <CmdkCommand.List className="max-h-64 overflow-y-auto p-2">
                <CmdkCommand.Empty className="px-4 py-8 text-xs text-center" style={{ color: 'var(--color-text-faint)' }}>
                  No commands found
                </CmdkCommand.Empty>
                {filtered.map((cmd) => (
                  <CmdkCommand.Item
                    key={cmd.id}
                    value={cmd.label}
                    onSelect={() => { cmd.action(); setOpen(false) }}
                    className="flex items-center justify-between px-3 py-2 rounded-md text-sm cursor-pointer"
                    style={{
                      color: 'var(--color-text)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(66,165,245,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    <span>{cmd.label}</span>
                    <span className="text-xs" style={{ color: 'var(--color-text-faint)' }}>
                      {cmd.shortcut}
                    </span>
                  </CmdkCommand.Item>
                ))}
              </CmdkCommand.List>
            </CmdkCommand>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
