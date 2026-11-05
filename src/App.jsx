import Corridor from './Corridor.jsx';

// Corridor = live PS1 background; the memory interface (public/memory.html) sits on a glass panel above it.
export default function App() {
  return (
    <main className="page">
      <Corridor interactive={false} />

      <section className="hero">
        <h1 className="title">
          <span>Hyper</span>
          <span><span className="echo">W</span>arp<span className="cursor" aria-hidden="true">_</span></span>
        </h1>
        <p className="tagline">Корпоративная память.<br />Спросить, запечь, сравнить.</p>
      </section>

      <div className="console">
        <iframe src="/memory.html?embed=1" title="Корпоративная память" />
      </div>
    </main>
  );
}
