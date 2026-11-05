import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

// ---------- corridor shape ----------
const SIDES = 6;
const APOTHEM = 3;
const SEG = 2.5;
const RINGS = 36;
const WALL_W = 2 * APOTHEM * Math.tan(Math.PI / SIDES);
const VERT_R = APOTHEM / Math.cos(Math.PI / SIDES);
const DUST = 900;
const DUST_RANGE = 20;

// ---------- PS1 look ----------
const LOW_H = 240;      // internal render height, upscaled with nearest-neighbour
const TEX = 128;        // texture resolution (tiles drawn in 512 space, scaled down)
const SNAP = 0.5;       // vertex snap grid = SNAP * low-res pixels (lower = more wobble)

// window openings per tile variant: [x, y, w, h] in texture pixels
const WINDOW_REGIONS = {
  0: [[90, 230, 332, 52]],
  2: [[180, 116, 152, 40]],
  3: [[114, 174, 284, 164]],
};

// ---------- procedural tile textures ----------
function drawTile(g, v, mode) {
  const isColor = mode === 'color';
  const isBump = mode === 'bump';
  const base   = isColor ? '#cfaa62' : isBump ? '#8a8a8a' : '#7a7a7a';
  const recess = isColor ? '#000000' : isBump ? '#000000' : '#ffffff';
  const seam   = isColor ? '#4a3016' : '#1a1a1a';
  const edge   = isColor ? '#d88a50' : '#e0e0e0';
  const grille = isColor ? '#6e6556' : '#262626';
  const glow   = isColor ? '#000000' : '#ffffff';

  g.fillStyle = base;
  g.fillRect(0, 0, 512, 512);

  g.strokeStyle = seam; g.lineWidth = 10; g.strokeRect(10, 10, 492, 492);
  g.strokeStyle = edge; g.lineWidth = 2;  g.strokeRect(20, 20, 472, 472);
  g.strokeStyle = seam; g.lineWidth = 4;  g.strokeRect(40, 40, 432, 432);

  [[62, 62], [450, 62], [62, 450], [450, 450]].forEach(([x, y]) => {
    g.fillStyle = seam; g.beginPath(); g.arc(x, y, 10, 0, Math.PI * 2); g.fill();
    g.fillStyle = edge; g.beginPath(); g.arc(x - 2, y - 2, 4, 0, Math.PI * 2); g.fill();
  });

  if (v === 0) {                        // dark slot with lip
    g.fillStyle = edge;   g.fillRect(84, 224, 344, 64);
    g.fillStyle = recess; g.fillRect(90, 230, 332, 52);
  } else if (v === 1) {                 // vent grille
    g.fillStyle = grille;
    for (let y = 296; y <= 364; y += 10) g.fillRect(118, y, 276, 4);
  } else if (v === 2) {                 // small window over grille
    g.fillStyle = recess; g.fillRect(180, 116, 152, 40);
    g.fillStyle = grille;
    for (let y = 300; y <= 380; y += 12) g.fillRect(180, y, 152, 5);
  } else {                              // large window panel
    g.fillStyle = seam; g.fillRect(100, 160, 312, 192);
    g.fillStyle = glow; g.fillRect(114, 174, 284, 164);
    g.fillStyle = edge; g.fillRect(114, 174, 284, 6);
  }

  g.strokeStyle = seam; g.lineWidth = 4; g.strokeRect(230, 404, 52, 52);
  g.fillStyle = edge;   g.fillRect(246, 420, 20, 20);
}

function makeTexture(v, mode) {
  const c = document.createElement('canvas');
  c.width = c.height = TEX;
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.scale(TEX / 512, TEX / 512);        // keep drawing code in 512 space; transform persists for star updates
  drawTile(g, v, mode);
  const t = new THREE.CanvasTexture(c);
  t.magFilter = THREE.NearestFilter;    // chunky texels, no smoothing
  t.minFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.anisotropy = 1;
  if (mode === 'color') t.encoding = THREE.sRGBEncoding;
  return t;
}

// Patch a built-in material: snap vertices to a coarse screen grid (PS1 jitter)
// and quantize output to 15-bit color with 4x4 ordered dithering.
const psxUniforms = { uSnap: { value: new THREE.Vector2(160, 120) } };
function ps1(mat) {
  mat.onBeforeCompile = shader => {
    shader.uniforms.uSnap = psxUniforms.uSnap;
    shader.vertexShader = 'uniform vec2 uSnap;\n' + shader.vertexShader.replace(
      '#include <project_vertex>',
      `#include <project_vertex>
      // skip vertices near/behind camera: dividing by tiny w explodes triangles
      if (gl_Position.w > 0.3) {
        gl_Position.xy = floor(gl_Position.xy / gl_Position.w * uSnap + 0.5) / uSnap * gl_Position.w;
      }`
    );
    shader.fragmentShader = `
      float bayer2(vec2 a) { a = floor(a); return fract(a.x / 2.0 + a.y * a.y * 0.75); }
      float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
    ` + shader.fragmentShader.replace(
      '#include <dithering_fragment>',
      `#include <dithering_fragment>
      gl_FragColor.rgb = floor(gl_FragColor.rgb * 31.0 + bayer4(gl_FragCoord.xy)) / 31.0;`
    );
  };
  return mat;
}

// warpKey: change it to fire a warp jump (instant high speed, then smooth slowdown)
export default function Corridor({ interactive = false, warpKey = 0 }) {
  const mountRef = useRef(null);
  const interactiveRef = useRef(interactive);
  const warpRef = useRef(warpKey);

  useEffect(() => { interactiveRef.current = interactive; }, [interactive]);
  useEffect(() => { warpRef.current = warpKey; }, [warpKey]);

  useEffect(() => {
    const mount = mountRef.current;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ---------- renderer / scene ----------
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(1);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.BasicShadowMap;   // hard, blocky shadows
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    mount.prepend(renderer.domElement);

    const FOG_COLOR = 0x4a2a16;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(FOG_COLOR);
    scene.fog = new THREE.FogExp2(FOG_COLOR, 0.05);

    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth / mount.clientHeight, 0.1, 200);
    camera.position.set(0, 0, 2);

    // render at ~240p, CSS stretches canvas to full screen (image-rendering: pixelated)
    function sizeLowRes() {
      const w = mount.clientWidth, h = mount.clientHeight;
      const lw = Math.round(LOW_H * w / h);
      renderer.setSize(lw, LOW_H, false);
      psxUniforms.uSnap.value.set(lw * 0.5 * SNAP, LOW_H * 0.5 * SNAP);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    sizeLowRes();

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;

    scene.add(new THREE.HemisphereLight(0xffb070, 0x5a3f22, 0.4));
    scene.add(new THREE.AmbientLight(0xffc080, 0.35));

    const keyLight = new THREE.PointLight(0xff9a40, 10, 24, 2);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.bias = -0.001;
    keyLight.shadow.normalBias = 0.04;
    keyLight.shadow.radius = 4;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 20;
    const farLight = new THREE.PointLight(0xffa050, 8, 14, 2);
    scene.add(keyLight, farLight);

    // ---------- materials ----------
    const tileSets = [0, 1, 2, 3].map(v => ({
      map: makeTexture(v, 'color'),
      bump: makeTexture(v, 'bump'),
      rough: makeTexture(v, 'rough'),
    }));

    const wallMats = tileSets.map(t => ps1(new THREE.MeshStandardMaterial({
      map: t.map,
      bumpMap: t.bump,
      bumpScale: 1.0,
      roughnessMap: t.rough,
      roughness: 1.0,
      metalness: 0.0,
      envMapIntensity: 0.3,
      side: THREE.DoubleSide,
    })));

    const plainMat = ps1(new THREE.MeshStandardMaterial({ color: 0x7a3e18, roughness: 0.7 }));
    const frameMat = ps1(new THREE.MeshStandardMaterial({ color: 0xb0601e, roughness: 0.6, envMapIntensity: 0.4 }));
    const stripMat = ps1(new THREE.MeshBasicMaterial({ color: 0xf08a3c }));
    const haloMat = ps1(new THREE.MeshBasicMaterial({
      color: 0xffa050, transparent: true, opacity: 0.1,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    const haloMat2 = ps1(new THREE.MeshBasicMaterial({
      color: 0xffd9a0, transparent: true, opacity: 0.04,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));

    // ---------- geometry ----------
    const wallGeo = new THREE.BoxGeometry(WALL_W, SEG, 0.6);
    const frameH = new THREE.BoxGeometry(WALL_W * 0.92, 0.14, 0.2);
    const frameV = new THREE.BoxGeometry(0.14, SEG * 0.92, 0.2);
    const stripGeo = new THREE.BoxGeometry(0.11, 0.11, SEG);
    const haloGeo = new THREE.BoxGeometry(0.34, 0.34, SEG);
    const halo2Geo = new THREE.BoxGeometry(0.9, 0.9, SEG);
    const up = new THREE.Vector3(0, 0, 1);

    function addFrame(wall) {
      [-SEG / 2 + 0.12, SEG / 2 - 0.12].forEach(y => {
        const m = new THREE.Mesh(frameH, frameMat);
        m.position.set(0, y, -0.4);
        m.castShadow = true;
        wall.add(m);
      });
      [-WALL_W / 2 + 0.14, WALL_W / 2 - 0.14].forEach(x => {
        const m = new THREE.Mesh(frameV, frameMat);
        m.position.set(x, 0, -0.4);
        m.castShadow = true;
        wall.add(m);
      });
    }

    // ---------- rings ----------
    const rings = [];
    for (let i = 0; i < RINGS; i++) {
      const ring = new THREE.Group();
      ring.position.z = -3 - i * SEG;

      for (let k = 0; k < SIDES; k++) {
        const theta = k * Math.PI * 2 / SIDES;
        const n = new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0);
        const t = new THREE.Vector3(-Math.sin(theta), Math.cos(theta), 0);

        const tileMat = wallMats[(Math.random() * wallMats.length) | 0];
        const wall = new THREE.Mesh(wallGeo, [plainMat, plainMat, plainMat, plainMat, plainMat, tileMat]);
        wall.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(t, up, n));
        wall.position.copy(n).multiplyScalar(APOTHEM + 0.3);
        wall.castShadow = true;
        wall.receiveShadow = true;
        addFrame(wall);
        ring.add(wall);

        const phi = theta + Math.PI / SIDES;
        const px = Math.cos(phi) * VERT_R;
        const py = Math.sin(phi) * VERT_R;
        [[stripGeo, stripMat], [haloGeo, haloMat], [halo2Geo, haloMat2]].forEach(([geo, mat]) => {
          const m = new THREE.Mesh(geo, mat);
          m.position.set(px, py, 0);
          ring.add(m);
        });
      }

      scene.add(ring);
      rings.push(ring);
    }

    // ---------- dust ----------
    const dustPos = new Float32Array(DUST * 3);
    for (let i = 0; i < DUST; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * (APOTHEM - 0.3);
      dustPos[i * 3] = Math.cos(a) * r;
      dustPos[i * 3 + 1] = Math.sin(a) * r;
      dustPos[i * 3 + 2] = -Math.random() * DUST_RANGE;
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    // fixed large bounds so the cloud is never frustum-culled as the camera moves
    dustGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -10), 1e4);
    const dustMat = ps1(new THREE.PointsMaterial({
      color: 0xffd9a0, size: 0.03, transparent: true, opacity: 0.6, fog: false,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(new THREE.Points(dustGeo, dustMat));

    // ---------- animated stars in the windows ----------
    const starList = {};
    for (const v in WINDOW_REGIONS) {
      starList[v] = WINDOW_REGIONS[v].map(([, , w, h]) =>
        Array.from({ length: 70 }, () => ({
          x: Math.random() * w,
          y: Math.random() * h,
          s: Math.random() < 0.1 ? 8 : 4,   // 512-space units: 1-2 texels at TEX=128
          a: 0.4 + Math.random() * 0.6,
          drift: 0.15 + Math.random() * 0.35,
        }))
      );
    }

    // ---------- flight state / input ----------
    const baseSpeed = reducedMotion ? 0.01 : 0.05;
    const boostSpeed = reducedMotion ? 0.05 : 0.3;
    let speed = baseSpeed;
    let boosting = false;
    let targetX = 0, targetY = 0;
    const camPos = { x: 0, y: 0 };

    function updateWindowStars(steps = 1) {
      for (const v in WINDOW_REGIONS) {
        const ctx = tileSets[v].map.image.getContext('2d');
        WINDOW_REGIONS[v].forEach(([x, y, w, h], ri) => {
          ctx.fillStyle = '#000000';
          ctx.fillRect(x, y, w, h);
          for (const s of starList[v][ri]) {
            s.y -= s.drift * 0.25 * (speed / 0.05) * steps;   // stars drift against the flight direction
            if (s.y < 0) s.y += h;
            ctx.fillStyle = `rgba(255,255,255,${s.a})`;
            ctx.fillRect(x + s.x, y + s.y, s.s, s.s);
          }
        });
        tileSets[v].map.needsUpdate = true;
      }
    }

    const onMove = e => {
      targetX = (e.clientX / window.innerWidth - 0.5) * 2.2;
      targetY = -(e.clientY / window.innerHeight - 0.5) * 1.6;
    };
    // boost only while exploring, so clicks on the title screen stay clicks
    const onDown = () => { if (interactiveRef.current) boosting = true; };
    const onUp = () => { boosting = false; };
    const onKeyDown = e => { if (e.code === 'Space' && interactiveRef.current) { e.preventDefault(); boosting = true; } };
    const onKeyUp = e => { if (e.code === 'Space') boosting = false; };
    const onResize = sizeLowRes;

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onResize);

    // ---------- loop ----------
    let time = 0;
    let raf = 0;
    let frame = 0;
    let lastWarp = warpRef.current;
    const WARP_SPEED = reducedMotion ? 0.15 : 1.8;   // units per frame at the peak of the jump
    const WARP_RAMP_FRAMES = 45;                      // ~0.75 s of acceleration before the peak
    let warpFrom = 0, warpFrame = -1;
    const BASE_FOV = camera.fov;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      time += 0.016;

      if (warpRef.current !== lastWarp) {             // warp jump: smooth ramp up, then long slowdown
        lastWarp = warpRef.current;
        warpFrom = speed;
        warpFrame = 0;
      }
      if (warpFrame >= 0 && warpFrame < WARP_RAMP_FRAMES) {
        const t = ++warpFrame / WARP_RAMP_FRAMES;
        const ease = t * t * (3 - 2 * t);             // smoothstep: gentle start, gentle peak
        speed = warpFrom + (WARP_SPEED - warpFrom) * ease;
        if (warpFrame === WARP_RAMP_FRAMES) warpFrame = -1;
      } else {
        const target = boosting ? boostSpeed : baseSpeed;
        // slower ease while far above cruise speed = long, smooth deceleration
        speed += (target - speed) * (speed > 0.4 ? 0.022 : 0.04);
      }
      camera.position.z -= speed;

      // FOV widens with speed: tunnel stretches during the jump, settles as we slow down
      const fov = BASE_FOV + Math.min(40, (speed - baseSpeed) * 26);
      if (Math.abs(camera.fov - fov) > 0.01) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
      }

      camPos.x += (targetX - camPos.x) * 0.04;
      camPos.y += (targetY - camPos.y) * 0.04;
      camera.position.x = camPos.x + Math.sin(time * 0.7) * 0.05;
      camera.position.y = camPos.y + Math.cos(time * 0.9) * 0.05;
      camera.lookAt(camPos.x * 0.5, camPos.y * 0.5, camera.position.z - 20);
      camera.rotation.z += Math.sin(time * 0.4) * 0.0005 - camPos.x * 0.001;

      keyLight.position.set(0, 0, camera.position.z - 14);
      farLight.position.set(0, 0, camera.position.z - 24);
      farLight.intensity = 8 + Math.sin(time * 23) * 1.2 + Math.sin(time * 7.3) * 2;

      for (const ring of rings) {
        if (ring.position.z > camera.position.z + SEG * 2) ring.position.z -= RINGS * SEG;
      }
      for (let i = 0; i < DUST; i++) {
        if (dustPos[i * 3 + 2] > camera.position.z + 2) dustPos[i * 3 + 2] -= DUST_RANGE;
      }
      dustGeo.attributes.position.needsUpdate = true;

      // re-upload 3 textures every 3rd frame, not every frame: less GPU stall, page stays smooth
      if (++frame % 3 === 0) updateWindowStars(3);
      renderer.render(scene, camera);
    };
    animate();

    // ---------- cleanup ----------
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);

      [wallGeo, frameH, frameV, stripGeo, haloGeo, halo2Geo, dustGeo].forEach(g => g.dispose());
      [...wallMats, plainMat, frameMat, stripMat, haloMat, haloMat2, dustMat].forEach(m => m.dispose());
      tileSets.forEach(t => { t.map.dispose(); t.bump.dispose(); t.rough.dispose(); });
      envTex.dispose();
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="corridor" ref={mountRef} aria-hidden="true">
      <div className="vignette" />
    </div>
  );
}
