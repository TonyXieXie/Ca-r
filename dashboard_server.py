"""Lightweight dashboard server for PPO training metrics.

Usage:
    python dashboard_server.py [--port 8765] [--runs-dir ./runs]

Open http://localhost:8765 in your browser.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# HTML page (embedded so no extra files are needed)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>PPO Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e0e0e0; --text2: #888; --accent: #6c8aff;
    --green: #4ade80; --red: #f87171; --yellow: #facc15;
    --blue: #60a5fa; --purple: #c084fc; --orange: #fb923c; --cyan: #22d3ee;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  
  /* Header */
  .header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; font-weight: 600; color: var(--accent); white-space: nowrap; }
  .header select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 13px; min-width: 280px; cursor: pointer; }
  .header select:focus { outline: none; border-color: var(--accent); }
  .header label { font-size: 13px; color: var(--text2); display: flex; align-items: center; gap: 6px; white-space: nowrap; }
  .header input[type="checkbox"] { accent-color: var(--accent); }
  .header input[type="number"] { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; width: 60px; font-size: 13px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; }
  .status-dot.paused { background: var(--yellow); }
  .status-dot.error { background: var(--red); }
  .header .info { font-size: 12px; color: var(--text2); margin-left: auto; }

  /* Stats bar */
  .stats-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 24px; display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; }
  .stat-item { display: flex; align-items: center; gap: 6px; }
  .stat-item .stat-label { color: var(--text2); }
  .stat-item .stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat-item .stat-value.good { color: var(--green); }
  .stat-item .stat-value.warn { color: var(--yellow); }
  .stat-item .stat-value.bad { color: var(--red); }

  /* Chart grid */
  .charts-container { padding: 16px 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .chart-card h3 { font-size: 13px; font-weight: 600; padding: 10px 14px 0; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
  .chart-card .chart-wrap { padding: 6px 10px 10px; height: 260px; }

  /* Group header */
  .group-header { grid-column: 1 / -1; font-size: 14px; font-weight: 700; color: var(--accent); padding: 16px 0 4px; border-bottom: 1px solid var(--border); margin-top: 8px; }
  .group-header:first-child { margin-top: 0; }

  /* No data */
  .no-data { text-align: center; padding: 80px 20px; color: var(--text2); font-size: 15px; }

  @media (max-width: 600px) {
    .charts-container { grid-template-columns: 1fr; padding: 12px; }
    .header { padding: 10px 12px; }
    .chart-card .chart-wrap { height: 220px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>🚗 PPO Training Dashboard</h1>
  <select id="fileSelect"><option value="">-- 选择训练日志 --</option></select>
  <label><input type="checkbox" id="autoRefresh" checked /> 自动刷新</label>
  <label>间隔 <input type="number" id="refreshInterval" value="5" min="1" max="60" /> 秒</label>
  <span class="status-dot" id="statusDot"></span>
  <span class="info" id="statusText">就绪</span>
</div>

<div class="stats-bar" id="statsBar"></div>

<div class="charts-container" id="chartsContainer">
  <div class="no-data">请从上方下拉菜单选择一个训练日志文件</div>
</div>

<script>
// ── Chart group definitions ──
const CHART_GROUPS = [
  {
    title: "📈 回报与回合",
    charts: [
      { keys: ["avg_return_20"], label: "最近20回合平均回报", color: "#4ade80" },
      { keys: ["rollout_reward_mean", "rollout_reward_std"], label: "Rollout 奖励均值/标准差", colors: ["#60a5fa", "#f87171"] },
      { keys: ["avg_ep_len_20"], label: "最近20回合平均长度", color: "#c084fc" },
      { keys: ["sps"], label: "训练速度 (steps/sec)", color: "#22d3ee" },
    ]
  },
  {
    title: "🧠 PPO 损失",
    charts: [
      { keys: ["policy_loss"], label: "策略损失 (Policy Loss)", color: "#fb923c" },
      { keys: ["value_loss"], label: "价值损失 (Value Loss)", color: "#f87171" },
      { keys: ["entropy"], label: "策略熵 (Entropy)", color: "#4ade80" },
      { keys: ["total_loss"], label: "总损失 (Total Loss)", color: "#c084fc" },
    ]
  },
  {
    title: "📐 PPO 诊断",
    charts: [
      { keys: ["clipfrac"], label: "Clip 比例", color: "#facc15", yMax: 1.0 },
      { keys: ["approx_kl"], label: "近似 KL 散度", color: "#fb923c" },
      { keys: ["explained_variance"], label: "解释方差", color: "#60a5fa", yMin: -0.5, yMax: 1.0 },
    ]
  },
  {
    title: "🔧 优化状态",
    charts: [
      { keys: ["grad_norm"], label: "梯度范数", color: "#f87171" },
      { keys: ["value_mean"], label: "价值均值", color: "#60a5fa" },
      { keys: ["learning_rate"], label: "学习率", color: "#4ade80" },
    ]
  },
  {
    title: "🏁 评估指标",
    charts: [
      { keys: ["eval_return_mean", "eval_return_std"], label: "评估回报均值/标准差", colors: ["#4ade80", "#f87171"] },
      { keys: ["eval_length_mean"], label: "评估回合平均长度", color: "#c084fc" },
    ]
  },
];

// ── State ──
let charts = {};        // key -> Chart instance
let currentFile = "";
let refreshTimer = null;
let lastLineCount = 0;

// ── API helpers ──
async function api(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

// ── Load file list ──
async function loadFileList() {
  try {
    const files = await api("/api/files");
    const sel = document.getElementById("fileSelect");
    const prev = sel.value;
    sel.innerHTML = '<option value="">-- 选择训练日志 --</option>';
    for (const f of files) {
      const opt = document.createElement("option");
      opt.value = f.path;
      opt.textContent = f.display;
      sel.appendChild(opt);
    }
    if (prev && files.some(f => f.path === prev)) sel.value = prev;
  } catch (e) {
    console.error("Failed to load file list:", e);
  }
}

// ── Load metrics data ──
async function loadMetrics() {
  if (!currentFile) return;
  try {
    const data = await api(`/api/data?file=${encodeURIComponent(currentFile)}`);
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusText");
    dot.className = "status-dot";
    txt.textContent = `已加载 ${data.total_lines} 条记录 · 最后更新 ${new Date().toLocaleTimeString()}`;

    if (data.total_lines !== lastLineCount) {
      lastLineCount = data.total_lines;
      renderCharts(data.records);
      renderStats(data.records);
    }
  } catch (e) {
    document.getElementById("statusDot").className = "status-dot error";
    document.getElementById("statusText").textContent = "加载失败: " + e.message;
  }
}

// ── Render stats bar ──
function renderStats(records) {
  const bar = document.getElementById("statsBar");
  if (!records.length) { bar.innerHTML = ""; return; }
  const last = records[records.length - 1];
  const items = [
    { label: "Update", value: last.update },
    { label: "Global Step", value: last.global_step?.toLocaleString() },
    { label: "平均回报", value: fmt(last.avg_return_20), cls: clsReturn(last.avg_return_20) },
    { label: "策略熵", value: fmt(last.entropy, 3) },
    { label: "Clip%", value: fmt(last.clipfrac * 100, 1) + "%", cls: clsClip(last.clipfrac) },
    { label: "KL", value: fmt(last.approx_kl, 4), cls: clsKL(last.approx_kl) },
    { label: "Expl.Var", value: fmt(last.explained_variance, 3), cls: clsEV(last.explained_variance) },
    { label: "Grad Norm", value: fmt(last.grad_norm, 2) },
    { label: "SPS", value: last.sps },
  ];
  bar.innerHTML = items.map(i =>
    `<div class="stat-item"><span class="stat-label">${i.label}:</span><span class="stat-value ${i.cls || ''}">${i.value ?? '—'}</span></div>`
  ).join("");
}

function fmt(v, d=1) { return v != null ? Number(v).toFixed(d) : null; }
function clsReturn(v) { if (v == null) return ''; return v > 200 ? 'good' : v > 0 ? '' : 'bad'; }
function clsClip(v) { if (v == null) return ''; return v > 0.4 ? 'bad' : v > 0.25 ? 'warn' : 'good'; }
function clsKL(v) { if (v == null) return ''; return v > 0.05 ? 'bad' : v > 0.02 ? 'warn' : 'good'; }
function clsEV(v) { if (v == null) return ''; return v > 0.8 ? 'good' : v > 0.3 ? 'warn' : 'bad'; }

// ── Render all charts ──
function renderCharts(records) {
  const container = document.getElementById("chartsContainer");
  container.innerHTML = "";

  // Use global_step as x-axis, fallback to update
  const xs = records.map(r => r.global_step ?? r.update);
  const hasEval = records.some(r => r.eval_return_mean != null);

  for (const group of CHART_GROUPS) {
    // Skip eval group if no eval data
    if (group.title.includes("评估") && !hasEval) continue;

    const header = document.createElement("div");
    header.className = "group-header";
    header.textContent = group.title;
    container.appendChild(header);

    for (const spec of group.charts) {
      // Check if any data exists for these keys
      const hasData = spec.keys.some(k => records.some(r => r[k] != null));
      if (!hasData) continue;

      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<h3>${spec.label}</h3><div class="chart-wrap"><canvas id="chart_${spec.keys.join('_')}"></canvas></div>`;
      container.appendChild(card);

      const colors = spec.colors || (spec.keys.length === 1 ? [spec.color] : spec.keys.map((_, i) => {
        const palette = ["#4ade80","#60a5fa","#f87171","#facc15","#c084fc","#fb923c","#22d3ee"];
        return palette[i % palette.length];
      }));

      const datasets = spec.keys.map((key, i) => {
        const ys = records.map(r => r[key] != null ? r[key] : null);
        return {
          label: key,
          data: ys,
          borderColor: colors[i],
          backgroundColor: colors[i] + "20",
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.2,
          fill: spec.keys.length === 1,
          spanGaps: true,
        };
      });

      const canvasId = `chart_${spec.keys.join('_')}`;
      const ctx = card.querySelector(`#${canvasId}`).getContext("2d");

      const yScale = {
        grid: { color: "rgba(255,255,255,0.06)" },
        ticks: { color: "#888", font: { size: 11 } },
      };
      if (spec.yMin != null) yScale.min = spec.yMin;
      if (spec.yMax != null) yScale.max = spec.yMax;

      const chart = new Chart(ctx, {
        type: "line",
        data: { labels: xs, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: spec.keys.length > 1, labels: { color: "#aaa", font: { size: 11 } } },
            tooltip: {
              backgroundColor: "#1a1d27",
              borderColor: "#2a2d3a",
              borderWidth: 1,
              titleColor: "#e0e0e0",
              bodyColor: "#aaa",
              callbacks: {
                title: (items) => `Step: ${items[0].label?.toLocaleString() ?? items[0].parsed.x}`,
              }
            },
          },
          scales: {
            x: {
              grid: { color: "rgba(255,255,255,0.04)" },
              ticks: { color: "#666", font: { size: 10 }, maxTicksLimit: 8,
                callback: function(v) { const val = this.getLabelForValue(v); return typeof val === 'number' ? val.toLocaleString() : val; }
              },
              title: { display: true, text: "Global Step", color: "#666", font: { size: 10 } },
            },
            y: yScale,
          },
        }
      });

      charts[spec.keys.join(",")] = chart;
    }
  }
}

// ── Auto refresh ──
function startAutoRefresh() {
  stopAutoRefresh();
  const interval = parseInt(document.getElementById("refreshInterval").value) || 5;
  refreshTimer = setInterval(() => {
    if (document.getElementById("autoRefresh").checked && currentFile) {
      loadMetrics();
      loadFileList(); // also refresh file list in case new runs appear
    }
  }, interval * 1000);
}

function stopAutoRefresh() {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
}

// ── Events ──
document.getElementById("fileSelect").addEventListener("change", (e) => {
  currentFile = e.target.value;
  lastLineCount = 0;
  if (currentFile) {
    loadMetrics();
  } else {
    document.getElementById("chartsContainer").innerHTML = '<div class="no-data">请从上方下拉菜单选择一个训练日志文件</div>';
    document.getElementById("statsBar").innerHTML = "";
  }
});

document.getElementById("autoRefresh").addEventListener("change", () => {
  const dot = document.getElementById("statusDot");
  if (document.getElementById("autoRefresh").checked) {
    dot.className = "status-dot";
    startAutoRefresh();
  } else {
    dot.className = "status-dot paused";
    stopAutoRefresh();
  }
});

document.getElementById("refreshInterval").addEventListener("change", () => {
  if (document.getElementById("autoRefresh").checked) startAutoRefresh();
});

// ── Init ──
loadFileList();
startAutoRefresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler: serves the HTML page and two JSON API endpoints."""

    runs_dir: Path = Path("./runs")

    def log_message(self, fmt, *args):
        # Quieter logging — only show errors
        if "200" not in str(args):
            super().log_message(fmt, *args)

    # ── Routing ──
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/files":
            self._api_files()
        elif path == "/api/data":
            params = parse_qs(parsed.query)
            filepath = params.get("file", [None])[0]
            self._api_data(filepath)
        else:
            self._respond(404, {"error": "not found"})

    # ── Endpoints ──
    def _serve_html(self):
        data = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_files(self):
        """List all metrics.jsonl files under runs_dir."""
        files = []
        for p in sorted(self.runs_dir.rglob("metrics.jsonl")):
            rel = p.relative_to(self.runs_dir)
            # Read first line to get a quick preview
            preview = ""
            try:
                with open(p, "r", encoding="utf-8") as f:
                    first = f.readline().strip()
                    if first:
                        obj = json.loads(first)
                        preview = f"  (update {obj.get('update','?')}, step {obj.get('global_step','?')})"
            except Exception:
                pass
            files.append({"path": str(rel), "display": str(rel) + preview})
        self._respond(200, files)

    def _api_data(self, filepath: str | None):
        """Read a specific metrics.jsonl and return all records."""
        if not filepath:
            self._respond(400, {"error": "missing 'file' parameter"})
            return

        target = (self.runs_dir / filepath).resolve()
        # Security: ensure it's under runs_dir
        if not str(target).startswith(str(self.runs_dir.resolve())):
            self._respond(403, {"error": "access denied"})
            return
        if not target.exists():
            self._respond(404, {"error": f"file not found: {filepath}"})
            return

        records = []
        total_lines = 0
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            self._respond(500, {"error": str(e)})
            return

        self._respond(200, {"total_lines": total_lines, "records": records})

    # ── Helpers ──
    def _respond(self, code: int, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="PPO Training Dashboard Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--runs-dir", type=str, default="./runs", help="Path to runs directory (default: ./runs)")
    args = parser.parse_args()

    runs = Path(args.runs_dir).resolve()
    DashboardHandler.runs_dir = runs

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"🚗 PPO Training Dashboard")
    print(f"   Runs directory: {runs}")
    print(f"   Open http://localhost:{args.port} in your browser")
    print(f"   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
