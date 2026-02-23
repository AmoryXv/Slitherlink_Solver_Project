"""
Flask 版 Slitherlink 可视化求解器
Slitherlink Visual Solver (Flask Edition)
用法 / Usage: python web_app.py  → http://localhost:5000
"""

import os, json, tempfile, base64, cv2, numpy as np
from flask import Flask, request, jsonify, render_template_string
from ocr_engine import SlitherlinkOCR
from puzzle_model import SlitherlinkPuzzle

app = Flask(__name__)
ocr = SlitherlinkOCR()

# ───────────────────── HTML 模板 ─────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Slitherlink Solver</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:  #0f1117;
    --card:#1a1d2e;
    --accent:#7c5cfc;
    --accent2:#38bdf8;
    --text:#e2e8f0;
    --muted:#94a3b8;
    --success:#22c55e;
    --danger:#ef4444;
    --border:#2d3148;
    --radius:12px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family:'Inter',system-ui,sans-serif;
    background:var(--bg); color:var(--text);
    min-height:100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);
    padding:28px 0; text-align:center;
    border-bottom:1px solid var(--border);
  }
  header h1 { font-size:1.8rem; font-weight:700; }
  header h1 span { color:var(--accent2); }
  header p { color:var(--muted); margin-top:4px; font-size:.9rem; }
  .sub-en { font-size:.78rem; color:#64748b; }

  /* ── Layout ── */
  .container { max-width:1280px; margin:24px auto; padding:0 20px; }
  .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }
  @media(max-width:960px){ .grid-3{grid-template-columns:1fr;} }

  /* ── Cards ── */
  .card {
    background:var(--card); border:1px solid var(--border);
    border-radius:var(--radius); padding:24px;
    box-shadow:0 4px 24px rgba(0,0,0,.25);
  }
  .card h2 {
    font-size:1.0rem; font-weight:600; margin-bottom:4px;
    display:flex; align-items:center; gap:8px;
  }
  .card .h2-en {
    font-size:.78rem; color:var(--muted); font-weight:400;
    margin-bottom:14px; padding-left:28px;
  }

  /* ── Upload ── */
  .upload-zone {
    border:2px dashed var(--border); border-radius:var(--radius);
    padding:36px 20px; text-align:center; cursor:pointer;
    transition:border-color .2s, background .2s;
    position: relative;
  }
  .upload-zone:hover,.upload-zone.dragover {
    border-color:var(--accent); background:rgba(124,92,252,.06);
  }
  .upload-zone input { position:absolute; inset:0; opacity:0; cursor:pointer; }
  .upload-zone .icon { font-size:2.2rem; margin-bottom:6px; }
  .upload-zone p { color:var(--muted); font-size:.82rem; }
  .upload-zone .filename { color:var(--accent2); font-weight:500; margin-top:6px; }

  /* ── Buttons ── */
  .btn {
    display:inline-flex; align-items:center; gap:6px;
    padding:10px 22px; border-radius:8px;
    border:none; cursor:pointer; font-size:.85rem; font-weight:600;
    transition:transform .12s, box-shadow .2s;
  }
  .btn:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(0,0,0,.3); }
  .btn:active { transform:translateY(0); }
  .btn-primary { background:linear-gradient(135deg,var(--accent),#6344e0); color:#fff; }
  .btn-success { background:linear-gradient(135deg,var(--success),#16a34a); color:#fff; }
  .btn-row { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }

  /* ── Table (matrix editor) ── */
  .matrix-wrap { overflow-x:auto; }
  table.matrix {
    border-collapse:collapse; width:100%;
    font-variant-numeric:tabular-nums;
  }
  table.matrix th { color:var(--muted); font-weight:500; font-size:.75rem; padding:4px 6px; }
  table.matrix td { padding:2px; }
  table.matrix input {
    width:38px; height:34px; text-align:center;
    background:#111425; border:1px solid var(--border);
    border-radius:6px; color:var(--text); font-size:.9rem;
    font-weight:600; outline:none;
    transition:border-color .15s;
  }
  table.matrix input:focus { border-color:var(--accent); }

  /* ── Canvas ── */
  #solutionCanvas {
    border:1px solid var(--border); border-radius:var(--radius);
    background:#111425; display:block; margin:0 auto;
    max-width:100%;
  }

  /* ── Status ── */
  .status {
    margin-top:12px; padding:10px 14px; border-radius:8px;
    font-size:.82rem; font-weight:500; display:none;
  }
  .status.ok   { display:block; background:rgba(34,197,94,.12); color:var(--success); border:1px solid rgba(34,197,94,.3); }
  .status.fail { display:block; background:rgba(239,68,68,.12); color:var(--danger);  border:1px solid rgba(239,68,68,.3); }
  .status.info { display:block; background:rgba(56,189,248,.10); color:var(--accent2); border:1px solid rgba(56,189,248,.25); }

  /* ── Spinner ── */
  .spinner { display:inline-block; width:18px; height:18px;
    border:2.5px solid rgba(255,255,255,.2);
    border-top-color:#fff; border-radius:50%;
    animation:spin .6s linear infinite; }
  @keyframes spin { to{transform:rotate(360deg);} }

  /* ── Preview ── */
  .preview-img {
    max-width:100%; border-radius:var(--radius);
    border:1px solid var(--border); display:block;
    margin:0 auto;
  }
  .preview-placeholder {
    text-align:center; padding:40px 16px;
    color:var(--muted); font-size:.82rem;
    border:1px dashed var(--border); border-radius:var(--radius);
    background:#111425;
  }
</style>
</head>
<body>

<header>
  <h1>🧩 Slitherlink <span>Auto Solver</span></h1>
  <p>上传谜题图片 → AI 自动识别 → 一键求解</p>
  <p class="sub-en">Upload puzzle image → AI recognition → One-click solve</p>
</header>

<div class="container">
  <!-- Step 1: Upload 上传 -->
  <div class="card" style="margin-bottom:24px;">
    <h2>📤 Step 1 — 上传谜题图片</h2>
    <div class="h2-en">Upload Puzzle Image</div>
    <div class="upload-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/png,image/jpeg">
      <div class="icon">🖼️</div>
      <p>点击选择 或 拖拽图片到此处<br><span style="font-size:.75rem">Click to select or drag & drop image here</span></p>
      <div class="filename" id="fileName"></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" id="btnRecognize" disabled>
        🔍 识别数字 / Recognise
      </button>
    </div>
    <div class="status" id="statusRecognize"></div>
  </div>

  <!-- Step 2 + 2b + 3 : 三栏布局 -->
  <div class="grid-3" id="resultArea" style="display:none;">
    <!-- 左栏：图片预览 -->
    <div class="card">
      <h2>🖼️ 原图预览</h2>
      <div class="h2-en">Image Preview</div>
      <div id="previewWrap">
        <div class="preview-placeholder">识别后显示原图<br>Image will appear after recognition</div>
      </div>
    </div>
    <!-- 中栏：矩阵校对 -->
    <div class="card">
      <h2>📊 Step 2 — 识别结果 & 校对</h2>
      <div class="h2-en">Recognition Result & Correction</div>
      <p style="color:var(--muted);font-size:.78rem;margin-bottom:10px;">
        修改错误数字（-1 = 空位, 0-3 = 数字）<br>
        <span style="font-size:.72rem">Edit incorrect digits (-1 = empty, 0-3 = clue), then click Solve.</span>
      </p>
      <div class="matrix-wrap" id="matrixWrap"></div>
      <div class="btn-row">
        <button class="btn btn-success" id="btnSolve">
          🚀 确认并求解 / Solve
        </button>
      </div>
      <div class="status" id="statusSolve"></div>
    </div>
    <!-- 右栏：求解结果 -->
    <div class="card">
      <h2>🎯 Step 3 — 求解结果</h2>
      <div class="h2-en">Solution</div>
      <canvas id="solutionCanvas" width="460" height="460"></canvas>
    </div>
  </div>
</div>

<script>
// ──────────── 全局变量 ────────────
let currentMatrix = null;
let rows = 0, cols = 0;

// ──────────── DOM ────────────
const fileInput     = document.getElementById('fileInput');
const dropZone      = document.getElementById('dropZone');
const fileName      = document.getElementById('fileName');
const btnRecognize  = document.getElementById('btnRecognize');
const btnSolve      = document.getElementById('btnSolve');
const resultArea    = document.getElementById('resultArea');
const matrixWrap    = document.getElementById('matrixWrap');
const previewWrap   = document.getElementById('previewWrap');
const statusR       = document.getElementById('statusRecognize');
const statusS       = document.getElementById('statusSolve');
const canvas        = document.getElementById('solutionCanvas');
const ctx           = canvas.getContext('2d');

// ──────────── 上传交互 ────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; onFileSelected(); }
});
fileInput.addEventListener('change', onFileSelected);

function onFileSelected() {
  const f = fileInput.files[0];
  if(!f) return;
  fileName.textContent = f.name;
  btnRecognize.disabled = false;
}

// ──────────── 识别 ────────────
btnRecognize.addEventListener('click', async () => {
  const f = fileInput.files[0];
  if(!f) return;
  btnRecognize.disabled = true;
  btnRecognize.innerHTML = '<span class="spinner"></span> 识别中 / Recognising...';
  setStatus(statusR, 'info', '⏳ 正在上传并识别，请稍候… Uploading & recognising…');

  const form = new FormData();
  form.append('image', f);

  try {
    const res = await fetch('/api/recognize', { method:'POST', body:form });
    const data = await res.json();
    if(!data.ok) throw new Error(data.error);

    currentMatrix = data.matrix;
    rows = data.rows;
    cols = data.cols;

    // 显示 warped 预览图
    if(data.warped_b64) {
      previewWrap.innerHTML = '<img class="preview-img" src="data:image/png;base64,' + data.warped_b64 + '" alt="Warped board preview 棋盘矫正预览">';
    }

    renderMatrixEditor(currentMatrix);
    resultArea.style.display = '';
    setStatus(statusR, 'ok', `✅ 识别成功 Recognition OK — ${rows}×${cols}`);
    clearCanvas();
  } catch(e) {
    setStatus(statusR, 'fail', '❌ ' + e.message);
  } finally {
    btnRecognize.disabled = false;
    btnRecognize.innerHTML = '🔍 识别数字 / Recognise';
  }
});

// ──────────── 矩阵编辑器 ────────────
function renderMatrixEditor(mat) {
  let h = '<table class="matrix"><tr><th></th>';
  for(let c=0;c<cols;c++) h += `<th>${c}</th>`;
  h += '</tr>';
  for(let r=0;r<rows;r++) {
    h += `<tr><th>${r}</th>`;
    for(let c=0;c<cols;c++) {
      const v = mat[r][c];
      h += `<td><input type="number" min="-1" max="3" value="${v}" data-r="${r}" data-c="${c}"></td>`;
    }
    h += '</tr>';
  }
  h += '</table>';
  matrixWrap.innerHTML = h;
}

function readMatrixFromEditor() {
  const mat = [];
  for(let r=0;r<rows;r++) {
    const row = [];
    for(let c=0;c<cols;c++) {
      const inp = matrixWrap.querySelector(`input[data-r="${r}"][data-c="${c}"]`);
      let v = parseInt(inp.value, 10);
      if(isNaN(v) || v < -1 || v > 3) v = -1;
      row.push(v);
    }
    mat.push(row);
  }
  return mat;
}

// ──────────── 求解 ────────────
btnSolve.addEventListener('click', async () => {
  const mat = readMatrixFromEditor();
  btnSolve.disabled = true;
  btnSolve.innerHTML = '<span class="spinner"></span> 求解中 / Solving...';
  setStatus(statusS, 'info', '⏳ 正在求解，请稍候… Solving…');

  try {
    const res = await fetch('/api/solve', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ matrix:mat, rows, cols })
    });
    const data = await res.json();
    if(!data.ok) throw new Error(data.error);

    drawSolution(mat, data.h_edges, data.v_edges);
    setStatus(statusS, 'ok', `✅ 求解成功 Solved! (${data.time})`);
  } catch(e) {
    setStatus(statusS, 'fail', '❌ ' + e.message);
  } finally {
    btnSolve.disabled = false;
    btnSolve.innerHTML = '🚀 确认并求解 / Solve';
  }
});

// ──────────── Canvas 画图 ────────────
function clearCanvas() {
  ctx.fillStyle = '#111425';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawSolution(clues, hEdges, vEdges) {
  const pad = 36;
  const cw = (canvas.width  - 2*pad) / cols;
  const ch = (canvas.height - 2*pad) / rows;
  clearCanvas();

  // 网格点
  ctx.fillStyle = '#475569';
  for(let r=0;r<=rows;r++)
    for(let c=0;c<=cols;c++) {
      ctx.beginPath();
      ctx.arc(pad+c*cw, pad+r*ch, 3, 0, Math.PI*2);
      ctx.fill();
    }

  // 数字
  ctx.font = `bold ${Math.min(cw,ch)*0.42}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for(let r=0;r<rows;r++)
    for(let c=0;c<cols;c++) {
      const v = clues[r][c];
      if(v < 0) continue;
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(v, pad+c*cw+cw/2, pad+r*ch+ch/2);
    }

  // 水平边
  for(let r=0;r<=rows;r++)
    for(let c=0;c<cols;c++) {
      const e = hEdges[r][c];
      if(e === 1) {
        ctx.strokeStyle = '#7c5cfc'; ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(pad+c*cw, pad+r*ch);
        ctx.lineTo(pad+(c+1)*cw, pad+r*ch);
        ctx.stroke();
      } else if(e === 2) {
        drawCross(pad+c*cw+cw/2, pad+r*ch, 5);
      }
    }

  // 垂直边
  for(let r=0;r<rows;r++)
    for(let c=0;c<=cols;c++) {
      const e = vEdges[r][c];
      if(e === 1) {
        ctx.strokeStyle = '#7c5cfc'; ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(pad+c*cw, pad+r*ch);
        ctx.lineTo(pad+c*cw, pad+(r+1)*ch);
        ctx.stroke();
      } else if(e === 2) {
        drawCross(pad+c*cw, pad+r*ch+ch/2, 5);
      }
    }
}

function drawCross(x, y, s) {
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x-s,y-s); ctx.lineTo(x+s,y+s); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x+s,y-s); ctx.lineTo(x-s,y+s); ctx.stroke();
}

// ──────────── Util ────────────
function setStatus(el, type, msg) {
  el.className = 'status ' + type;
  el.textContent = msg;
}
</script>
</body>
</html>
"""

# ───────────────────── API ─────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """接收上传图片 → OCR 识别 → 返回数字矩阵 + warped 预览"""
    if 'image' not in request.files:
        return jsonify(ok=False, error='未收到图片文件 / No image file received')

    f = request.files['image']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    f.save(tmp.name)
    tmp.close()

    try:
        warped, matrix, r, c, _dbg = ocr.recognize_board_auto(tmp.name)
        mat_list = matrix.tolist()
        mat_list = [[int(v) for v in row] for row in mat_list]

        # 把 warped 图编码成 base64 以供前端预览
        warped_b64 = ''
        if warped is not None:
            _, buf = cv2.imencode('.png', warped)
            warped_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify(ok=True, matrix=mat_list, rows=r, cols=c,
                       warped_b64=warped_b64)
    except Exception as e:
        return jsonify(ok=False, error=str(e))
    finally:
        try: os.unlink(tmp.name)
        except: pass


@app.route('/api/solve', methods=['POST'])
def api_solve():
    """接收校正后的矩阵 → 求解 → 返回边数据"""
    data = request.get_json(force=True)
    mat  = data.get('matrix')
    r    = data.get('rows')
    c    = data.get('cols')

    if not mat or not r or not c:
        return jsonify(ok=False, error='参数不完整 / Incomplete parameters')

    try:
        import time as _t
        puzzle = SlitherlinkPuzzle(r, c, mat)
        puzzle.apply_basic_rules()

        t0 = _t.time()
        solved = puzzle.solve_backtracking()
        elapsed = _t.time() - t0

        if not solved:
            return jsonify(ok=False,
                           error='无解 — 请检查数字是否正确 / No solution — please verify digits')

        return jsonify(
            ok=True,
            h_edges=puzzle.h_edges.tolist(),
            v_edges=puzzle.v_edges.tolist(),
            time=f'{elapsed:.2f}s'
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e))


if __name__ == '__main__':
    print('\n  Slitherlink Solver Web UI')
    print('  http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
