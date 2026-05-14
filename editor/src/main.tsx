import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import WelcomeScreen from './screens/WelcomeScreen'
import EditorScreen from './screens/EditorScreen'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<WelcomeScreen />} />
        <Route path="/editor" element={<EditorScreen />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
