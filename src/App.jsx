import { useEffect, useState } from 'react';
import Corridor from './Corridor.jsx';

// Two screens over one live PS1 corridor:
//   start screen - big "Hyper Warp_" title + "Explore the space"
//   main screen  - title shrinks to the left, memory interface (public/memory.html) on a glass panel
export default function App() {
  const [started, setStarted] = useState(false);
  const [warp, setWarp] = useState(0);   // bump = corridor warp jump
  const go = next => { setStarted(next); setWarp(w => w + 1); };

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') setStarted(s => { if (s) setWarp(w => w + 1); return false; }); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <main className={started ? 'page is-started' : 'page'}>
      <Corridor interactive={false} warpKey={warp} />

      <section className="hero">
        <h1 className="title">
          <span>Hyper</span>
          <span><span className="echo">W</span>arp<span className="cursor" aria-hidden="true">_</span></span>
        </h1>
        <div className="taglines">
          <p className="tagline tagline-start" aria-hidden={started}>Somewhere<br />beyond ordinary.</p>
          <p className="tagline tagline-main" aria-hidden={!started}>Corporate memory.<br />Ask, bake, compare.</p>
        </div>
        <button className="explore" onClick={() => go(true)} tabIndex={started ? -1 : 0} aria-hidden={started}>
          Explore the space <span aria-hidden="true">→</span>
        </button>
      </section>

      <button className="back" onClick={() => go(false)} tabIndex={started ? 0 : -1} aria-hidden={!started}>
        ← Back
      </button>

      <div className="console" aria-hidden={!started}>
        {/* mounted once so state (answers, bakes) survives going back to the start screen */}
        <iframe src="/memory.html?embed=1" title="Corporate Memory" tabIndex={started ? 0 : -1} />
      </div>
    </main>
  );
}
