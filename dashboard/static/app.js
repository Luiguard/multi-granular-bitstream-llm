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
    const worldTokens = data.total_world_tokens || 0;
    const tokenStr = worldTokens >= 1000000 ? `${(worldTokens / 1000000).toFixed(2)}M` : worldTokens.toLocaleString();
    const corpusShards = data.total_corpus_shards || data.shards_processed || 1397;
    updateElement('shards-count-value', `${tokenStr} (${corpusShards} Shards)`);

    // Live Loss Metrics (Aktuell & Ø Durchschnitt)
    const currentLoss = data.current_loss !== undefined ? Number(data.current_loss).toFixed(4) : '--';
    updateElement('current-loss-value', currentLoss);

    let avgLossVal = data.average_loss;
    if (avgLossVal === undefined && data.loss_history && data.loss_history.length > 0) {
      avgLossVal = data.loss_history.reduce((a, b) => a + b, 0) / data.loss_history.length;
    }
    const avgLoss = avgLossVal !== undefined ? Number(avgLossVal).toFixed(4) : '--';
    updateElement('average-loss-value', avgLoss);
    updateElement('chart-loss-stats', `Aktuell: ${currentLoss} | Ø: ${avgLoss}`);

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

    // 3. Live Benchmark Metrics
    if (data.validation_ppl !== undefined) {
      updateElement('bench-ppl-val', Number(data.validation_ppl).toFixed(2));
    }
    if (data.mmlu_score !== undefined) {
      updateElement('bench-mmlu-val', `${Number(data.mmlu_score).toFixed(1)}%`);
    }
    if (data.inference_tps !== undefined) {
      updateElement('bench-tps-val', `${Number(data.inference_tps).toFixed(1)} T/s`);
    }
    if (data.compression_ratio !== undefined) {
      updateElement('bench-compress-val', `${Number(data.compression_ratio).toFixed(2)}x`);
    }

    // 4. Loss Chart History
    if (data.loss_history && data.loss_history.length > 0) {
      lossHistory = data.loss_history;
      drawChart();
    }

    // 5. Dynamic Training Knowledge Graph
    if (data.training_graph) {
      renderTrainingGraph(data.training_graph, data.active_knowledge_node);
    }

  } catch (err) {
    console.error('Metrics fetch error:', err);
  }
}

function formatTokens(count) {
  if (!count || count === 0) return '0 Tokens';
  if (count >= 1000000000) return (count / 1000000000).toFixed(2) + ' Mrd. Tokens';
  if (count >= 1000000) return (count / 1000000).toFixed(2) + ' Mio. Tokens';
  if (count >= 1000) return (count / 1000).toFixed(1) + 'k Tokens';
  return count + ' Tokens';
}

function renderTrainingGraph(graphData, activeNodeName) {
  if (!graphData || !graphData.nodes) return;

  const container = document.getElementById('training-graph-container');
  const masteryBadge = document.getElementById('graph-mastery-badge');
  const activeTag = document.getElementById('active-node-tag');
  const alertBox = document.getElementById('remediation-alert-box');
  const alertText = document.getElementById('remediation-text');

  if (masteryBadge) {
    masteryBadge.textContent = `${graphData.mastered_count || 0}/${graphData.total_nodes || 6} Mastered · ${graphData.active_count || 1} Aktiv`;
  }
  if (activeTag && activeNodeName) {
    activeTag.textContent = `Aktiv: ${activeNodeName}`;
  }

  if (alertBox && alertText) {
    if (graphData.has_active_remediation && graphData.active_remediation_text) {
      alertBox.style.display = 'flex';
      alertText.textContent = graphData.active_remediation_text;
    } else {
      alertBox.style.display = 'none';
    }
  }

  if (!container) return;

  container.innerHTML = '';
  graphData.nodes.forEach(node => {
    const box = document.createElement('div');
    const statusLower = (node.status || 'locked').toLowerCase();
    box.className = `graph-node-box ${statusLower}`;

    let badgeIcon = '🔒';
    if (node.status === 'ACTIVE') badgeIcon = '⚡';
    if (node.status === 'MASTERED') badgeIcon = '✅';

    const learnedTokens = (node.sample_count || 0) * 7168;

    box.innerHTML = `
      <div class="node-header">
        <span class="node-title">${node.name}</span>
        <span class="node-badge ${statusLower}">${badgeIcon} ${node.status}</span>
      </div>
      <p class="node-desc">${node.description}</p>
      <div class="node-token-bar">
        <span class="node-token-label">📚 Gelernt:</span>
        <span class="node-token-value">${formatTokens(learnedTokens)} (${node.sample_count || 0} Batches)</span>
      </div>
      <div class="node-stats">
        <span>Shards: ${node.total_shards}</span>
        <span>Loss: <strong class="node-loss">${node.moving_loss ? Number(node.moving_loss).toFixed(2) : '--'}</strong></span>
        ${node.remediation_boost > 1.05 ? `<span class="node-boost">Boost: ×${node.remediation_boost}</span>` : `<span>Target: ≤${node.mastery_threshold}</span>`}
      </div>
    `;
    container.appendChild(box);
  });
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

// Autonomous Learning Events Fetcher
async function fetchAutonomousLearning() {
  const container = document.getElementById('autonomous-events-list');
  const statusEl = document.getElementById('auto-learner-status');
  if (!container) return;

  try {
    const res = await fetch('/api/autonomous_learning');
    if (!res.ok) return;
    const data = await res.json();

    if (statusEl) {
      statusEl.textContent = data.status === 'active' ? '🟢 Aktiv (Live Daemon)' : 'Schläft';
    }

    if (!data.events || data.events.length === 0) {
      container.innerHTML = '<div style="font-size: 0.82rem; color: var(--text-muted);">Überwache Wissensknoten auf Wissenslücken (Loss &gt; 5.8)...</div>';
      return;
    }

    container.innerHTML = '';
    data.events.slice(-5).reverse().forEach(ev => {
      const card = document.createElement('div');
      card.style.cssText = 'background: hsla(222,47%,14%,0.8); border: 1px solid var(--border-glass); border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; display: flex; flex-direction: column; gap: 4px;';
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-weight: 600; color: var(--primary);">🧠 Lücke in: ${escapeHtml(ev.node_name)}</span>
          <span style="color: var(--text-muted); font-size: 0.72rem; font-family: var(--font-mono);">${ev.timestamp}</span>
        </div>
        <div style="color: var(--text-main); font-size: 0.78rem;">
          🔍 <strong>Recherchiertes Thema:</strong> <em>${escapeHtml(ev.researched_topic)}</em> (${ev.sources_count} Quellen)
        </div>
        <div style="display: flex; gap: 8px; font-size: 0.72rem; color: #a855f7; font-family: var(--font-mono);">
          <span>💾 Shard: ${ev.generated_shard}</span>
          <span>⚡ +${formatTokens(ev.tokens_generated)}</span>
          <span>Trigger Loss: ${Number(ev.trigger_loss).toFixed(2)}</span>
        </div>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    // ignore
  }
}

// Cognitive Heartbeat & Stream of Consciousness Fetcher
async function fetchCognitiveHeartbeat() {
  const thoughtsContainer = document.getElementById('cognitive-thoughts-list');
  const timelineContainer = document.getElementById('autobiographical-timeline-list');
  const heartbeatStatus = document.getElementById('heartbeat-status');
  if (!thoughtsContainer || !timelineContainer) return;

  try {
    const res = await fetch('/api/cognitive_heartbeat');
    if (!res.ok) return;
    const data = await res.json();

    if (heartbeatStatus) {
      heartbeatStatus.textContent = data.status === 'active' ? '💓 12s Takt (Aktiv)' : 'Pausiert';
    }

    // Render Thoughts
    if (!data.thoughts || data.thoughts.length === 0) {
      thoughtsContainer.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted);">Initiiere ersten internen Reflexionszyklus...</div>';
    } else {
      thoughtsContainer.innerHTML = '';
      data.thoughts.slice(-4).reverse().forEach(th => {
        const item = document.createElement('div');
        item.style.cssText = 'background: hsla(222,47%,12%,0.9); border-left: 3px solid var(--primary); border-radius: 4px; padding: 6px 10px; font-size: 0.76rem; display: flex; flex-direction: column; gap: 3px;';
        
        let passesHtml = '';
        if (th.passes) {
          th.passes.forEach(p => {
            passesHtml += `<div style="color: var(--text-muted); font-size: 0.72rem; line-height: 1.2;">• ${escapeHtml(p)}</div>`;
          });
        }

        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; color: var(--primary);">#${th.thought_id} ${escapeHtml(th.domain)}</span>
            <span style="color: #22c55e; font-size: 0.7rem; font-family: var(--font-mono);">${(th.confidence * 100).toFixed(1)}% Konfidenz</span>
          </div>
          ${passesHtml}
          <div style="color: var(--text-muted); font-size: 0.68rem; text-align: right; font-family: var(--font-mono);">${th.timestamp}</div>
        `;
        thoughtsContainer.appendChild(item);
      });
    }

    // Render Timeline
    if (!data.timeline || data.timeline.length === 0) {
      timelineContainer.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-muted);">Konsolidiere Meilensteine im 18-Bit Bitstream...</div>';
    } else {
      timelineContainer.innerHTML = '';
      data.timeline.slice(-4).reverse().forEach(ep => {
        const item = document.createElement('div');
        item.style.cssText = 'background: hsla(270,50%,12%,0.9); border-left: 3px solid #a855f7; border-radius: 4px; padding: 6px 10px; font-size: 0.76rem; display: flex; flex-direction: column; gap: 3px;';
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; color: #c084fc;">${escapeHtml(ep.title)}</span>
            <span style="color: var(--text-muted); font-size: 0.68rem; font-family: var(--font-mono);">${ep.timestamp}</span>
          </div>
          <div style="color: var(--text-main); font-size: 0.72rem;">${escapeHtml(ep.reflection_summary)}</div>
          <div style="display: flex; gap: 8px; font-size: 0.68rem; color: #a855f7; font-family: var(--font-mono);">
            <span>⚡ ${escapeHtml(ep.hebbian_weight_boost)}</span>
            <span>Knoten: ${ep.active_graph_nodes}</span>
          </div>
        `;
        timelineContainer.appendChild(item);
      });
    }

  } catch (err) {
    // ignore
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Initial Setup
resizeCanvas();
fetchMetrics();
fetchAutonomousLearning();
fetchCognitiveHeartbeat();
setInterval(fetchMetrics, 1000);
setInterval(fetchAutonomousLearning, 3000);
setInterval(fetchCognitiveHeartbeat, 3000);
