// Stub — full waveform + beat grid timeline coming in MVP 1
import { useRef, useEffect } from 'react'
import { useEditor } from '../../stores/editor'

export default function Timeline() {
  const { playhead, visibleRange, timelineZoom } = useEditor()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (e.metaKey || e.ctrlKey) {
        // zoom
        useEditor.getState().setZoom(timelineZoom * (e.deltaY < 0 ? 1.1 : 0.9))
      } else {
        // scroll
        const delta = e.deltaY > 0 ? 2 : -2
        useEditor.getState().panRange(delta)
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [timelineZoom])

  return (
    <div ref={ref} className="h-full w-full relative" style={{ background: 'var(--color-bg)' }}>
      {/* Grid lines placeholder */}
      <div className="absolute inset-0 flex items-center justify-center" style={{ color: 'var(--color-text-faint)' }}>
        <div className="text-center">
          <div className="text-4xl mb-3">🎼</div>
          <div className="text-sm font-medium">Timeline</div>
          <div className="text-xs mt-1" style={{ color: 'var(--color-text-dim)' }}>
            Audio waveform · Beat grid · Note lanes
          </div>
          <div className="text-xs mt-2" style={{ color: 'var(--color-text-faint)' }}>
            scroll=pan · ⌘scroll=zoom · snap {useEditor.getState().snapGrid}
          </div>
        </div>
      </div>

      {/* Playhead line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 pointer-events-none"
        style={{
          left: `${((playhead - visibleRange[0]) / (visibleRange[1] - visibleRange[0])) * 100}%`,
          background: 'var(--color-cyan)',
          boxShadow: '0 0 8px rgba(0,229,255,0.4)',
          opacity: playhead > 0 ? 1 : 0,
        }}
      />
    </div>
  )
}
