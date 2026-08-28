// Live Telemetry & Loss Chart Handler

let lossHistory = [];
const canvas = document.getElementById('lossChart');
const ctx = canvas ? canvas.getContext('2d') : null;

function resizeCanvas() {
  if (!canvas) return;
  canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
  drawChart();
}
window.addEventListener('resize', resizeCanvas);

function drawChart() {
  if (!canvas || !ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (lossHistory.length < 2) {
    ctx.fillStyle = '#64748b';
    ctx.font = `${14 * window.devicePixelRatio}px JetBrains Mono`;
    ctx.textAlign = 'center';
    ctx.fillText('Sammle Loss-Datenpunkte...', w / 2, h / 2);
    return;
  }

  const padding = 35 * window.devicePixelRatio;
  const graphW = w - padding * 2;
  const graphH = h - padding * 2;

  const minLoss = Math.min(...lossHistory) * 0.95;
  const maxLoss = Math.max(...lossHistory) * 1.05;
  const range = maxLoss - minLoss || 1;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1 * window.devicePixelRatio;
  for (let i = 0; i <= 4; i++) {
    const y = padding + (graphH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(w - padding, y);
    ctx.stroke();

    const val = (maxLoss - (i / 4) * range).toFixed(2);
    ctx.fillStyle = '#64748b';
    ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono`;
    ctx.textAlign = 'right';
    ctx.fillText(val, padding - 8 * window.devicePixelRatio, y + 4 * window.devicePixelRatio);
  }

  // Draw smooth gradient curve
  ctx.beginPath();
  const stepX = graphW / (lossHistory.length - 1);

  lossHistory.forEach((val, idx) => {
    const x = padding + idx * stepX;
    const y = padding + graphH - ((val - minLoss) / range) * graphH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  // Stroke line
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 3 * window.devicePixelRatio;
  ctx.shadowColor = '#06b6d4';
  ctx.shadowBlur = 10 * window.devicePixelRatio;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Fill gradient underneath
  ctx.lineTo(padding + graphW, padding + graphH);
  ctx.lineTo(padding, padding + graphH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padding, 0, padding + graphH);
  grad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
  grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
  ctx.fillStyle = grad;
  ctx.fill();
}

function updateElement(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function updateStyle(id, prop, val) {
  const el = document.getElementById(id);
  if (el) el.style[prop] = val;
}

async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();

    // 1. Progress & ETA
    updateElement('epoch-step-label', `Epoche ${data.epoch || 1}/${data.max_epochs || 1} · Step ${(data.step || 0).toLocaleString()} / ${(data.total_steps || 0).toLocaleString()}`);
    const percent = Math.min(100, Math.max(0, data.progress_percent || 0)).toFixed(1);
    updateElement('percent-label', `${percent}%`);
    updateStyle('progress-bar-fill', 'width', `${percent}%`);

    const statusTag = document.getElementById('training-status-tag');
    if (statusTag) {
      if (percent >= 100.0 || data.status === 'COMPLETED') {
        statusTag.textContent = 'ABGESCHLOSSEN (100%)';
        statusTag.className = 'status-tag completed';
      } else {
        statusTag.textContent = 'TRAINING AKTIV';
        statusTag.className = 'status-tag active';
      }
    }

    updateElement('eta-value', data.eta_str || '--:-- min');
    updateElement('throughput-value', `${(data.tokens_per_sec || 0).toLocaleString()} Tokens/s`);
    updateElement('current-loss-value', Number(data.current_loss || 0).toFixed(4));
    updateElement('shards-count-value', `${data.shards_processed || 24} Shards`);

    // 2. Hardware Telemetry
    if (data.gpu_vram_used_gb !== undefined) {
      updateElement('vram-text', `${data.gpu_vram_used_gb.toFixed(1)} / ${data.gpu_vram_total_gb.toFixed(1)} GB`);
      const vramPct = (data.gpu_vram_used_gb / data.gpu_vram_total_gb) * 100;
      updateStyle('vram-bar', 'width', `${Math.min(100, vramPct)}%`);
    }

    if (data.gpu_temp_c !== undefined) {
      updateElement('gpu-temp-text', `${data.gpu_temp_c}°C · ${data.gpu_util_pct || 0}% Util`);
      updateStyle('gpu-temp-bar', 'width', `${data.gpu_util_pct || 0}%`);
    }

    if (data.ram_used_gb !== undefined) {
      updateElement('ram-text', `${data.ram_used_gb.toFixed(1)} / ${data.ram_total_gb.toFixed(1)} GB`);
      const ramPct = (data.ram_used_gb / data.ram_total_gb) * 100;
      updateStyle('ram-bar', 'width', `${Math.min(100, ramPct)}%`);
    }

    if (data.cpu_util_pct !== undefined) {
      updateElement('cpu-text', `${data.cpu_util_pct}%`);
      updateStyle('cpu-bar', 'width', `${data.cpu_util_pct}%`);
    }

    // 3. Loss Chart History
    if (data.loss_history && data.loss_history.length > 0) {
      lossHistory = data.loss_history;
      drawChart();
    }

  } catch (err) {
    console.error('Metrics fetch error:', err);
  }
}

// Preset Helper
window.setPrompt = function(text) {
  const input = document.getElementById('prompt-input');
  if (input) {
    input.value = text;
    input.focus();
  }
};

// Interactive Prompt Test
const genBtn = document.getElementById('btn-generate');
if (genBtn) {
  genBtn.addEventListener('click', async () => {
    const input = document.getElementById('prompt-input');
    const prompt = input ? input.value.trim() : '';
    const outputEl = document.getElementById('generation-output');
    if (!prompt || !outputEl) return;

    outputEl.textContent = '⚡ Generiere über Viterbi-Bitstream...';
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt, max_tokens: 40 })
      });
      const result = await res.json();
      outputEl.textContent = result.text || 'Keine Antwort erhalten.';
    } catch (e) {
      outputEl.textContent = 'Fehler bei der Generierung: ' + e;
    }
  });
}

// Initial Setup
resizeCanvas();
fetchMetrics();
setInterval(fetchMetrics, 1000);
