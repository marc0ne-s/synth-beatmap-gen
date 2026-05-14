import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useProject } from '../stores/project'

export default function WelcomeScreen() {
  const navigate = useNavigate()
  const { clear, setProject } = useProject()
  const [loadingFile, setLoadingFile] = useState(false)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files.length > 0) {
      void tryLoadSynth(e.dataTransfer.files[0])
    }
  }, [navigate])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      void tryLoadSynth(e.target.files[0])
    }
  }, [navigate])

  const tryLoadSynth = async (file: File) => {
    setLoadingFile(true)
    try {
      // MVP 1: verify it's a .synth (ZIP) and extract metadata
      if (!file.name.endsWith('.synth')) {
        alert('Please drop a .synth file')
        setLoadingFile(false)
        return
      }
      const arrayBuffer = await file.arrayBuffer()
      const bytes = new Uint8Array(arrayBuffer)
      // Check ZIP magic number
      const isZip = bytes[0] === 0x50 && bytes[1] === 0x4B
      if (!isZip) {
        alert('Not a valid .synth file (must be a ZIP archive)')
        setLoadingFile(false)
        return
      }

      // Parse with JSZip (lazy import for size)
      const JSZip = (await import('jszip')).default
      const zip = await JSZip.loadAsync(bytes)
      const metaEntry = zip.file('synthriderz.meta.json') || zip.file('beatmap.meta.bin')
      if (!metaEntry) {
        alert('No beatmap metadata found in .synth file')
        setLoadingFile(false)
        return
      }

      let raw: any
      const isBin = !!zip.file('beatmap.meta.bin')
      if (isBin) {
        const data = await zip.file('beatmap.meta.bin')!.async('text')
        raw = JSON.parse(data)
      } else {
        const data = await zip.file('synthriderz.meta.json')!.async('text')
        raw = JSON.parse(data)
      }

      // Extract available difficulties
      const difficulties = Object.keys(raw.Track || {}).filter(
        (d) => d !== 'Length' && raw.Track[d] && Object.keys(raw.Track[d]).length > 0
      )

      setProject({
        path: null,
        title: raw.Name || file.name.replace('.synth', ''),
        artist: raw.Author || 'Unknown',
        bpm: raw.BPM || 120,
        duration: raw.Duration || 0,
        difficulty: difficulties.includes('Expert') ? 'Expert' : difficulties[0] as any,
        notes: [],
        dirty: false,
      })

      navigate('/editor')
    } catch (err) {
      console.error(err)
      alert('Failed to load .synth file: ' + String(err))
    } finally {
      setLoadingFile(false)
    }
  }

  const handleNewBlank = () => {
    clear()
    navigate('/editor')
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-10 p-8"
    >
      { /* Brand */ }
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
          Cross-platform beatmap editor for VR rhythm
        </p>
      </motion.div>

      { /* Drop Zone */ }
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.15, duration: 0.5 }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="w-full max-w-xl rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer relative"
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
        {loadingFile && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-2xl z-10">
            <div className="text-sm font-medium">Loading…</div>
          </div>
        )}
        <div className="text-5xl mb-4">🎵</div>
        <h3 className="text-xl font-semibold mb-2">Drop a .synth map file</h3>
        <p style={{ color: 'var(--color-text-dim)' }} className="mb-6">Drag & drop or browse to open a map</p>
        <div className="flex gap-3 justify-center">
          <input
            type="file"
            accept=".synth"
            id="synth-file-input"
            onChange={handleFileInput}
            className="hidden"
          />
          <label
            htmlFor="synth-file-input"
            className="px-5 py-2.5 rounded-lg font-medium text-sm transition-all hover:brightness-110 cursor-pointer"
            style={{
              background: 'rgba(66, 165, 245, 0.12)',
              border: '1px solid rgba(66, 165, 245, 0.2)',
              color: 'var(--color-blue)',
            }}
          >Browse Files</label>
          <button
            onClick={handleNewBlank}
            className="px-5 py-2.5 rounded-lg font-medium text-sm transition-all hover:brightness-110"
            style={{
              background: 'rgba(0, 229, 160, 0.08)',
              border: '1px solid rgba(0, 229, 160, 0.15)',
              color: 'var(--color-green)',
            }}
          >New Blank Map</button>
        </div>
      </motion.div>

      { /* Footer shortcuts */ }
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="flex gap-6 text-xs"
        style={{ color: 'var(--color-text-faint)' }}
      >
        <span>⌘K Command Palette</span>
        <span>⌘O Open Map…</span>
        <span>⌘, Settings</span>
      </motion.div>
    </div>
  )
}
