// Preview3D — Three.js viewport for in-game-like map preview
// Stub for MVP 1. Full implementation with note/rail/wall rendering coming.
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function Preview3D() {
  const gridRef = useRef<THREE.Group>(null)

  useFrame((_, delta) => {
    // Gentle rotation for the empty viewport
    if (gridRef.current) {
      gridRef.current.rotation.y += delta * 0.05
    }
  })

  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 5, 5]} intensity={0.5} />

      {/* Reference grid */}
      <group ref={gridRef}>
        <gridHelper args={[8, 16, '#3a4d6b', '#1a2540']} />
        {/* Center marker */}
        <mesh position={[0, 0.01, 0]}>
          <ringGeometry args={[0.15, 0.2, 32]} />
          <meshBasicMaterial color="#42a5f5" side={2} />
        </mesh>
      </group>

      {/* Playfield bounds */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2, 1, 8)]} />
        <lineBasicMaterial color="#1a2540" transparent opacity={0.3} />
      </lineSegments>
    </>
  )
}
