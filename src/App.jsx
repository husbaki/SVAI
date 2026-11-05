import { useEffect, useState } from 'react';
import Corridor from './Corridor.jsx';

export default function App() {
  const [exploring, setExploring] = useState(false);

  // Esc leaves the corridor and brings the title back
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') setExploring(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <main className={exploring ? 'page is-exploring' : 'page'}>
      <Corridor interactive={exploring} />

      <section className="hero" aria-hidden={exploring}>
        <h1 className="title">
          <span>Hyper</span>
          <span><span className="echo">W</span>arp<span className="cursor" aria-hidden="true">_</span></span>
        </h1>
        <p className="tagline">Somewhere<br />beyond ordinary.</p>
        <button className="explore" onClick={() => setExploring(true)} tabIndex={exploring ? -1 : 0}>
          Explore the space <span aria-hidden="true">↓</span>
        </button>
      </section>

      <p className="footnote">A space to get lost in.</p>

      <div className="controls" aria-hidden={!exploring}>
        <p>Мышь или тач — рулить. Зажать — ускорение.</p>
        <button onClick={() => setExploring(false)} tabIndex={exploring ? 0 : -1}>Назад</button>
      </div>
    </main>
  );
}
