"""Dependency-free web viewer for Cascad trace bundles."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def run_viewer(
    runs_dir: str | Path = "runs",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the Cascad visual trace viewer."""
    root = Path(runs_dir).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/runs":
                self._send_json(list_runs(root))
                return
            if parsed.path == "/api/run":
                query = parse_qs(parsed.query)
                run_id = query.get("id", [""])[0]
                self._send_json(load_run(root, run_id))
                return
            self.send_error(404, "Not found")

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send_json(self, value: object) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, value: str, content_type: str) -> None:
            body = value.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Cascad viewer: {url}")
    print(f"Runs directory: {root}")
    server.serve_forever()


def list_runs(root: Path) -> list[dict[str, object]]:
    """List available trace bundles."""
    if not root.exists():
        return []
    runs = []
    for child in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        trace_path = child / "trace.json"
        if not metrics_path.exists() or not trace_path.exists():
            continue
        metrics = _read_json(metrics_path)
        runs.append(
            {
                "id": child.name,
                "mtime": child.stat().st_mtime,
                "metrics": metrics,
            }
        )
    return runs


def load_run(root: Path, run_id: str) -> dict[str, object]:
    """Load one run bundle for the viewer."""
    run_dir = (root / run_id).resolve()
    if root not in run_dir.parents and run_dir != root:
        return {"error": "Invalid run id"}
    if not run_dir.exists():
        return {"error": "Run not found"}

    trace = _read_json(run_dir / "trace.json")
    metrics = _read_json(run_dir / "metrics.json")
    dot = (run_dir / "causal_graph.dot").read_text(encoding="utf-8")
    events = trace.get("events", [])
    observations = trace.get("observations", [])
    edges = _parse_dot_edges(dot)
    nodes = sorted({event.get("node_id", "") for event in events} | {item for edge in edges for item in (edge["source"], edge["target"])})
    affected = set(metrics.get("affected_nodes", []))
    return {
        "id": run_id,
        "trace": trace,
        "metrics": metrics,
        "events": events,
        "observations": observations,
        "graph": {
            "nodes": [
                {
                    "id": node,
                    "affected": node in affected,
                    "event_count": len([event for event in events if event.get("node_id") == node]),
                }
                for node in nodes
                if node
            ],
            "edges": edges,
        },
    }


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dot_edges(dot: str) -> list[dict[str, object]]:
    edges = []
    for line in dot.splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        source = left.strip().strip('"')
        target_part, _, attr_part = right.partition("[")
        target = target_part.strip().strip('"')
        label = "causal"
        color = "gray"
        if attr_part:
            attrs = attr_part.rstrip("];")
            for attr in attrs.split(","):
                key, _, value = attr.partition("=")
                if key.strip() == "label":
                    label = value.strip().strip('"')
                if key.strip() == "color":
                    color = value.strip().strip('"')
        edges.append({"source": source, "target": target, "label": label, "color": color})
    return edges


INDEX_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cascad Viewer</title>
  <style>
    :root {
      --font: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --radius: 18px;
      --shadow: 0 18px 45px rgba(15, 23, 42, .16);
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg:#080b12; --shell:#0d1320; --panel:#111827; --panel-2:#162033;
      --soft:#1f2937; --line:#263244; --muted:#94a3b8; --text:#e5edf8;
      --text-strong:#f8fafc; --blue:#60a5fa; --blue-2:#2563eb; --green:#22c55e;
      --red:#fb7185; --orange:#f59e0b; --purple:#a78bfa; --chip:#172033;
      --graph-a:#111827; --graph-b:#0b1020;
    }
    :root[data-theme="light"] {
      color-scheme: light;
      --bg:#f5f7fb; --shell:#eef3fb; --panel:#ffffff; --panel-2:#f8fafc;
      --soft:#eef2f7; --line:#d9e2ef; --muted:#64748b; --text:#1e293b;
      --text-strong:#0f172a; --blue:#2563eb; --blue-2:#1d4ed8; --green:#16a34a;
      --red:#e11d48; --orange:#d97706; --purple:#7c3aed; --chip:#eef4ff;
      --graph-a:#f8fbff; --graph-b:#eaf1fb;
      --shadow: 0 18px 45px rgba(30, 41, 59, .10);
    }
    * { box-sizing: border-box; }
    body {
      margin:0; font-family:var(--font); background:linear-gradient(135deg, var(--shell), var(--bg));
      color:var(--text); min-height:100vh;
    }
    button, .run, .node, .copy { cursor:pointer; }
    button {
      background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:12px;
      padding:9px 12px; font-weight:650; transition:transform .16s ease, border-color .16s ease, background .16s ease, color .16s ease;
    }
    button:hover { transform:translateY(-1px); border-color:var(--blue); color:var(--text-strong); }
    button:active { transform:translateY(0); }
    header {
      min-height:72px; display:flex; align-items:center; justify-content:space-between; gap:18px;
      padding:14px clamp(14px, 2.5vw, 28px); border-bottom:1px solid var(--line);
      background:color-mix(in srgb, var(--panel) 85%, transparent); backdrop-filter:blur(12px);
      position:sticky; top:0; z-index:10;
    }
    h1 { font-size:19px; margin:0; letter-spacing:.2px; color:var(--text-strong); }
    h2 { font-size:14px; margin:0 0 12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
    .brand { display:flex; align-items:center; gap:12px; min-width:0; }
    .logo {
      width:38px; height:38px; border-radius:14px; display:grid; place-items:center;
      background:linear-gradient(135deg, var(--blue), var(--purple)); color:white; font-weight:900;
      box-shadow:0 12px 30px rgba(37, 99, 235, .28);
    }
    .subtitle { margin-top:2px; color:var(--muted); font-size:12px; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; justify-content:flex-end; }
    main {
      display:grid; grid-template-columns: minmax(260px, 330px) minmax(430px, 1fr) minmax(300px, 380px);
      gap:16px; padding:16px; min-height:calc(100vh - 72px);
    }
    aside, section.panel {
      background:color-mix(in srgb, var(--panel) 94%, transparent); border:1px solid var(--line);
      border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden;
    }
    .sidebar, .details { padding:16px; min-height:0; }
    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }
    .run-list { display:grid; gap:10px; max-height:calc(100vh - 170px); overflow:auto; padding-right:2px; }
    .run {
      border:1px solid var(--line); border-radius:16px; padding:12px; background:linear-gradient(180deg, var(--panel), var(--panel-2));
      transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .run:hover { transform:translateY(-2px); border-color:var(--blue); box-shadow:0 12px 28px rgba(37, 99, 235, .14); }
    .run.active { border-color:var(--blue); box-shadow:0 0 0 2px color-mix(in srgb, var(--blue) 28%, transparent) inset; }
    .run-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .run-id { font-family:var(--mono); font-size:12px; color:var(--text-strong); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .muted { color:var(--muted); font-size:12px; }
    .status-dot { width:9px; height:9px; border-radius:50%; background:var(--green); box-shadow:0 0 0 4px color-mix(in srgb, var(--green) 18%, transparent); flex:0 0 auto; }
    .run-stats { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
    .metrics { display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap:12px; padding:14px; border-bottom:1px solid var(--line); }
    .metric {
      background:linear-gradient(180deg, var(--panel-2), var(--panel)); border:1px solid var(--line);
      border-radius:16px; padding:13px; min-width:0;
    }
    .metric strong { display:block; font-size:24px; margin-top:4px; color:var(--text-strong); overflow:hidden; text-overflow:ellipsis; }
    .metric .hint { color:var(--muted); font-size:11px; margin-top:4px; }
    .graph-header { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px; border-bottom:1px solid var(--line); }
    .graph-title { min-width:0; }
    .graph-title strong { display:block; color:var(--text-strong); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    #graphWrap { height: min(58vh, 620px); min-height:420px; position:relative; }
    svg { width:100%; height:100%; background:radial-gradient(circle at 20% 0%, var(--graph-a), var(--graph-b) 62%); cursor:grab; touch-action:none; }
    svg:active { cursor:grabbing; }
    .edge { stroke:#7b8794; stroke-width:2.2; marker-end:url(#arrow); opacity:.9; }
    .edge.affected { stroke:var(--red); }
    .node { filter:drop-shadow(0 10px 16px rgba(0,0,0,.18)); }
    .node:hover circle { stroke-width:4; }
    .node circle { fill:var(--blue-2); stroke:#bfdbfe; stroke-width:2.4; }
    .node.affected circle { fill:#9f1239; stroke:var(--red); }
    .node.selected circle { stroke:var(--text-strong); stroke-width:4; }
    .node text { fill:white; font-size:12px; font-weight:750; text-anchor:middle; pointer-events:none; }
    .edge-label { fill:var(--muted); font-size:11px; text-anchor:middle; }
    .details { overflow:auto; max-height:calc(100vh - 104px); }
    pre {
      white-space:pre-wrap; word-break:break-word; background:var(--panel-2); border:1px solid var(--line);
      border-radius:14px; padding:12px; font-size:12px; line-height:1.5; color:var(--text);
    }
    .timeline { padding:14px; border-top:1px solid var(--line); max-height:300px; overflow:auto; }
    .event {
      border-left:4px solid #7b8794; padding:10px 12px; margin-bottom:9px; background:var(--panel-2);
      border-radius:0 14px 14px 0; transition:transform .16s ease, background .16s ease;
    }
    .event:hover { transform:translateX(2px); }
    .event.error { border-left-color:var(--red); }
    .pill {
      display:inline-flex; align-items:center; border:1px solid var(--line); color:var(--muted); background:var(--chip);
      border-radius:999px; padding:3px 8px; font-size:11px; margin:0 5px 5px 0; max-width:100%;
    }
    .empty {
      border:1px dashed var(--line); border-radius:16px; padding:18px; color:var(--muted);
      background:var(--panel-2); text-align:center;
    }
    .copy { color:var(--blue); background:none; border:0; padding:0; font:inherit; font-weight:750; }
    .toast {
      position:fixed; right:16px; bottom:16px; background:var(--text-strong); color:var(--bg);
      padding:10px 12px; border-radius:12px; box-shadow:var(--shadow); opacity:0; transform:translateY(8px);
      transition:opacity .2s ease, transform .2s ease; z-index:30; pointer-events:none;
    }
    .toast.show { opacity:1; transform:translateY(0); }
    @media (max-width: 1180px) {
      main { grid-template-columns: 300px 1fr; }
      .details { grid-column:1 / -1; max-height:none; }
      .metrics { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
    }
    @media (max-width: 760px) {
      header { align-items:flex-start; flex-direction:column; }
      .actions { width:100%; justify-content:stretch; }
      .actions button { flex:1; }
      main { grid-template-columns:1fr; padding:10px; gap:10px; }
      .run-list { max-height:260px; }
      .metrics { grid-template-columns:1fr 1fr; padding:10px; gap:10px; }
      .graph-header { align-items:flex-start; flex-direction:column; }
      #graphWrap { height:430px; min-height:360px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo">C</div>
      <div>
        <h1>Cascad Trace Viewer</h1>
        <div class="subtitle">Causal error propagation, metrics and intervention evidence.</div>
      </div>
    </div>
    <div class="actions">
      <button onclick="loadRuns()">Refresh</button>
      <button onclick="resetGraph()">Reset graph</button>
      <button id="themeButton" onclick="toggleTheme()">Dark mode</button>
    </div>
  </header>
  <main>
    <aside class="sidebar">
      <div class="toolbar">
        <h2>Runs</h2>
        <div class="muted" id="runsMeta">Loading...</div>
      </div>
      <div class="run-list" id="runs"></div>
    </aside>
    <section class="panel">
      <div class="metrics" id="metrics"></div>
      <div class="graph-header">
        <div class="graph-title">
          <span class="muted">Selected trace</span>
          <strong id="activeRunTitle">No run selected</strong>
        </div>
        <div>
          <button onclick="copyRunId()">Copy run ID</button>
        </div>
      </div>
      <div id="graphWrap"><svg id="graph"></svg></div>
      <div class="timeline" id="timeline"></div>
    </section>
    <aside class="details">
      <h2>Selection</h2>
      <div id="details" class="empty">Select a run or graph node.</div>
    </aside>
  </main>
  <div id="toast" class="toast">Copied</div>
  <script>
    let current = null;
    let selectedNode = null;
    let pan = {x:0, y:0, scale:1};
    let dragging = null;
    let positions = {};

    async function loadRuns() {
      const runs = await fetch('/api/runs').then(r => r.json());
      document.getElementById('runsMeta').textContent = `${runs.length} run(s)`;
      const el = document.getElementById('runs');
      el.innerHTML = '';
      if (!runs.length) {
        el.innerHTML = '<div class="empty">No Cascad run found. Point the viewer to your agent_data/cascad_runs directory.</div>';
        document.getElementById('metrics').innerHTML = '';
        document.getElementById('timeline').innerHTML = '';
        document.getElementById('activeRunTitle').textContent = 'No run selected';
        return;
      }
      runs.forEach((run, index) => {
        const div = document.createElement('div');
        div.className = 'run' + (index === 0 ? ' active' : '');
        div.title = run.id;
        const m = run.metrics || {};
        div.innerHTML = `
          <div class="run-top">
            <div class="run-id">${escapeHtml(formatId(run.id))}</div>
            <span class="status-dot"></span>
          </div>
          <div class="muted" style="margin-top:4px">${formatDate(run.mtime)}</div>
          <div class="run-stats">
            <span class="pill">depth ${safe(m.propagation_depth)}</span>
            <span class="pill">breadth ${safe(m.propagation_breadth)}</span>
            <span class="pill">delay ${formatSeconds(m.propagation_delay)}</span>
          </div>`;
        div.onclick = () => selectRun(run.id, div);
        el.appendChild(div);
        if (index === 0) selectRun(run.id, div);
      });
    }

    async function selectRun(id, div) {
      document.querySelectorAll('.run').forEach(item => item.classList.remove('active'));
      if (div) div.classList.add('active');
      current = await fetch(`/api/run?id=${encodeURIComponent(id)}`).then(r => r.json());
      selectedNode = null;
      positions = {};
      document.getElementById('activeRunTitle').textContent = current.error ? current.error : formatId(id);
      document.getElementById('activeRunTitle').title = id;
      renderMetrics();
      renderGraph();
      renderTimeline();
      document.getElementById('details').innerHTML = `<pre>${escapeHtml(JSON.stringify(current.metrics, null, 2))}</pre>`;
    }

    function renderMetrics() {
      const m = current.metrics || {};
      const metrics = [
        ['Propagation depth', safe(m.propagation_depth), 'steps crossed'],
        ['Propagation delay', formatSeconds(m.propagation_delay), 'time to visible effect'],
        ['Propagation breadth', safe(m.propagation_breadth), 'affected components'],
        ['Memory amp.', Number(m.memory_amplification_factor || 0).toFixed(3), 'memory amplification factor']
      ];
      document.getElementById('metrics').innerHTML = metrics.map(([k,v,h]) => `<div class="metric"><span class="muted">${k}</span><strong>${v}</strong><div class="hint">${h}</div></div>`).join('');
    }

    function renderGraph() {
      const svg = document.getElementById('graph');
      const graph = current.graph || {nodes:[], edges:[]};
      const w = svg.clientWidth || 900, h = svg.clientHeight || 520;
      const cx = w / 2, cy = h / 2;
      const r = Math.min(w, h) * 0.32;
      graph.nodes.forEach((node, i) => {
        if (!positions[node.id]) {
          const angle = (Math.PI * 2 * i / Math.max(1, graph.nodes.length)) - Math.PI / 2;
          positions[node.id] = {x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r};
        }
      });
      svg.innerHTML = `<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#7b8794"></path></marker></defs>`;
      const viewport = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      viewport.setAttribute('transform', `translate(${pan.x},${pan.y}) scale(${pan.scale})`);
      svg.appendChild(viewport);
      graph.edges.forEach(edge => {
        const a = positions[edge.source], b = positions[edge.target];
        if (!a || !b) return;
        const line = el('line', {x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:`edge ${edge.color === 'red' ? 'affected' : ''}`});
        viewport.appendChild(line);
        viewport.appendChild(el('text', {x:(a.x+b.x)/2, y:(a.y+b.y)/2 - 6, class:'edge-label'}, edge.label));
      });
      graph.nodes.forEach(node => {
        const p = positions[node.id];
        const g = el('g', {class:`node ${node.affected ? 'affected' : ''} ${selectedNode === node.id ? 'selected' : ''}`, transform:`translate(${p.x},${p.y})`});
        g.appendChild(el('circle', {r:34}));
        g.appendChild(el('text', {y:5}, compactNode(node.id)));
        g.onmousedown = evt => { dragging = node.id; evt.stopPropagation(); };
        g.onclick = evt => { selectedNode = node.id; showNode(node.id); renderGraph(); evt.stopPropagation(); };
        viewport.appendChild(g);
      });
      svg.onwheel = evt => { evt.preventDefault(); pan.scale = Math.max(.35, Math.min(2.4, pan.scale + (evt.deltaY < 0 ? .08 : -.08))); renderGraph(); };
      svg.onmousemove = evt => {
        if (!dragging) return;
        positions[dragging] = {x:(evt.offsetX - pan.x)/pan.scale, y:(evt.offsetY - pan.y)/pan.scale};
        renderGraph();
      };
      svg.onmouseup = () => dragging = null;
      svg.onmouseleave = () => dragging = null;
    }

    function resetGraph() { pan = {x:0,y:0,scale:1}; positions = {}; renderGraph(); }

    function showNode(id) {
      const events = (current.events || []).filter(e => e.node_id === id);
      const obs = (current.observations || []).filter(o => o.node_id === id);
      document.getElementById('details').innerHTML = `<div><span class="pill">${escapeHtml(id)}</span><span class="pill">${events.length} events</span><span class="pill">${obs.length} observations</span></div><pre>${escapeHtml(JSON.stringify({events, observations: obs}, null, 2))}</pre>`;
    }

    function renderTimeline() {
      const events = current.events || [];
      document.getElementById('timeline').innerHTML = '<h2>Timeline</h2>' + (events.length ? events.map(e => {
        const cls = String(e.kind).includes('error') || JSON.stringify(e.payload || {}).toLowerCase().includes('error') ? 'event error' : 'event';
        return `<div class="${cls}"><span class="pill">${escapeHtml(e.node_id)}</span><span class="pill">${escapeHtml(String(e.kind))}</span><div class="muted">${escapeHtml(JSON.stringify(e.payload || {})).slice(0, 280)}</div></div>`;
      }).join('') : '<div class="empty">No event recorded for this run.</div>');
    }

    function el(name, attrs, text) {
      const node = document.createElementNS('http://www.w3.org/2000/svg', name);
      Object.entries(attrs || {}).forEach(([k,v]) => node.setAttribute(k, v));
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
    function safe(v) { return v === undefined || v === null || v === '' ? 'n/a' : v; }
    function formatSeconds(v) { return v === undefined || v === null ? 'n/a' : `${Number(v || 0).toFixed(4)}s`; }
    function formatId(id) {
      id = String(id || '');
      if (id.length <= 18) return id;
      return `${id.slice(0, 8)}…${id.slice(-6)}`;
    }
    function compactNode(id) {
      id = String(id || '');
      return id.length > 11 ? `${id.slice(0, 5)}…${id.slice(-4)}` : id;
    }
    function formatDate(ts) {
      if (!ts) return 'unknown time';
      return new Date(ts * 1000).toLocaleString([], {dateStyle:'medium', timeStyle:'short'});
    }
    function copyRunId() {
      if (!current || !current.id) return;
      navigator.clipboard?.writeText(current.id);
      const toast = document.getElementById('toast');
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1400);
    }
    function applyTheme(theme) {
      document.documentElement.dataset.theme = theme;
      localStorage.setItem('cascad-theme', theme);
      document.getElementById('themeButton').textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    }
    function toggleTheme() {
      applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    }

    applyTheme(localStorage.getItem('cascad-theme') || 'dark');
    loadRuns();
  </script>
</body>
</html>
"""
