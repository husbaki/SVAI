import { useEffect, useRef, useState } from 'react';
import Corridor from './Corridor.jsx';

// Corridor = live PS1 background; the memory interface (public/memory.html) sits on a glass panel above it.
export default function App() {
  const [consoleHidden, setConsoleHidden] = useState(true);
  const [started, setStarted] = useState(false);
  const [hasOpened, setHasOpened] = useState(false);
  const [warp, setWarp] = useState(0);
  const frameRef = useRef(null);
  const showButtonRef = useRef(null);
  const hideButtonRef = useRef(null);

  const hideConsole = () => {
    setConsoleHidden(true);
    setStarted(false);
    setWarp(w => w + 1);
    requestAnimationFrame(() => showButtonRef.current?.focus());
  };
  const showConsole = () => {
    setConsoleHidden(false);
    setStarted(true);
    setHasOpened(true);
    setWarp(w => w + 1);
    requestAnimationFrame(() => hideButtonRef.current?.focus());
  };

  useEffect(() => {
    const onKey = event => {
      if (event.key === 'Escape') hideConsole();
    };
    const onMessage = event => {
      if (event.origin === window.location.origin &&
          event.source === frameRef.current?.contentWindow &&
          event.data?.type === 'memory:hide') hideConsole();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('message', onMessage);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('message', onMessage);
    };
  }, []);

  return (
    <main className={started ? "page is-started" : "page"} data-memory-open={!consoleHidden}>
      <Corridor interactive={false} warpKey={warp} />

      <section className="hero">
        <h1 className="title">
          <span>Hyper</span>
          <span><span className="echo">W</span>arp<span className="cursor" aria-hidden="true">_</span></span>
        </h1>
        <div className="taglines">
          <p className="tagline tagline-start" aria-hidden={started}>Somewhere<br />beyond ordinary.</p>
          <p className="tagline tagline-main" aria-hidden={!started}>Understand your code.<br />See the source.</p>
        </div>
        <button className="explore" onClick={showConsole} hidden={hasOpened} tabIndex={started ? -1 : 0}>
          Explore the space <span aria-hidden="true">→</span>
        </button>
      </section>

      <div className="console" id="memory-console" hidden={consoleHidden}>
        <div className="console-header">
          <span>Ask your codebase</span>
          <button ref={hideButtonRef} className="console-toggle" onClick={hideConsole}
            aria-controls="memory-console" aria-expanded="true" title="Hide window (Esc)">
            Hide window <span aria-hidden="true">−</span>
          </button>
        </div>
        <iframe ref={frameRef} src="/memory.html?embed=1" title="Corporate memory" />
      </div>
      <button ref={showButtonRef} className="console-toggle console-reopen" hidden={!consoleHidden || !hasOpened}
        onClick={showConsole} aria-controls="memory-console" aria-expanded="false">
        Show memory <span aria-hidden="true">↗</span>
      </button>
    </main>
  );
}
