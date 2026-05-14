// Playhead — playback cursor overlay
import { useEditor } from '../../stores/editor'

export default function Playhead() {
  const { isPlaying, playhead, togglePlay } = useEditor()

  return (
    <div className="absolute bottom-4 left-4 right-4 flex items-center gap-3"
      style={{ pointerEvents: 'none' }}
    >
      <button
        onClick={togglePlay}
        className="w-8 h-8 rounded-lg flex items-center justify-center text-sm pointer-events-auto"
        style={{
          background: 'rgba(66,165,245,0.1)',
          border: '1px solid rgba(66,165,245,0.2)',
          color: 'var(--color-blue)',
        }}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>
      <span className="text-xs font-mono" style={{ color: 'var(--color-text-dim)' }}>
        {(playhead).toFixed(2)}s
      </span>
    </div>
  )
}
