// InspectorPanel — right sidebar for object properties
export default function InspectorPanel() {
  return (
    <div className="h-full flex flex-col p-4"
      style={{ background: 'var(--color-bg)' }}
    >
      <div className="text-xs font-semibold uppercase tracking-widest mb-4"
        style={{ color: 'var(--color-text-faint)' }}
      >Inspector</div>

      <div className="flex-1 flex flex-col items-center justify-center gap-3"
        style={{ color: 'var(--color-text-faint)' }}
      >
        <div className="text-3xl">📋</div>
        <div className="text-sm font-medium text-center">No selection</div>
        <div className="text-xs text-center" style={{ color: 'var(--color-text-dim)' }}>
          Click a note, rail, or wall<br />to edit its properties
        </div>
      </div>

      {/* Quick stats */}
      <div className="mt-auto pt-4 border-t" style={{ borderColor: 'var(--color-border)' }}>
        <div className="text-xs" style={{ color: 'var(--color-text-faint)' }}>
          <div className="flex justify-between mb-1">
            <span>Notes</span><span className="font-mono">0</span>
          </div>
          <div className="flex justify-between mb-1">
            <span>Rails</span><span className="font-mono">0</span>
          </div>
          <div className="flex justify-between mb-1">
            <span>Walls</span><span className="font-mono">0</span>
          </div>
          <div className="flex justify-between">
            <span>Duration</span><span className="font-mono">0:00</span>
          </div>
        </div>
      </div>
    </div>
  )
}
