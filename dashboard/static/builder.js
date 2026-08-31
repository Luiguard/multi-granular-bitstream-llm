/**
 * Bitstream AI Architecture Studio | JavaScript Controller
 */

let CURRENT_SPECS = {};
let PRESETS = {};

// DOM Elements
const presetsContainer = document.getElementById('presets-container');
const inputModelName = document.getElementById('input-model-name');
const selectDModel = document.getElementById('select-d-model');
const selectLayers = document.getElementById('select-layers');
const selectHeads = document.getElementById('select-heads');
const selectExperts = document.getElementById('select-experts');
const selectTopK = document.getElementById('select-top-k');
const selectSeqLen = document.getElementById('select-seq-len');
const selectVocab = document.getElementById('select-vocab');

const rangeRlvr = document.getElementById('range-rlvr');
const rangeGuard = document.getElementById('range-guard');
const rangeCuriosity = document.getElementById('range-curiosity');

const valRlvr = document.getElementById('val-rlvr');
const valGuard = document.getElementById('val-guard');
const valCuriosity = document.getElementById('val-curiosity');

const statTotalParams = document.getElementById('stat-total-params');
const statTotalSub = document.getElementById('stat-total-sub');
const statActiveParams = document.getElementById('stat-active-params');
const statActiveSub = document.getElementById('stat-active-sub');
const statSparsity = document.getElementById('stat-sparsity');

const feasibilityBanner = document.getElementById('feasibility-banner');
const feasibilityIcon = document.getElementById('feasibility-icon');
const feasibilityTitle = document.getElementById('feasibility-title');
const feasibilityDesc = document.getElementById('feasibility-desc');

const memTrainGpu = document.getElementById('mem-train-gpu');
const memTrainRam = document.getElementById('mem-train-ram');
const memInferVram = document.getElementById('mem-infer-vram');
const memInferHybridRam = document.getElementById('mem-infer-hybrid-ram');

const codeOutput = document.getElementById('code-output');
const btnCopyCode = document.getElementById('btn-copy-code');
const btnGenerateModel = document.getElementById('btn-generate-model');
const btnExportRecipe = document.getElementById('btn-export-recipe');
const builderToast = document.getElementById('builder-toast');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  setupSliderListeners();
  setupFormListeners();
  setupDomainCards();
  await loadPresets();
  triggerRecalculation();
});

function setupSliderListeners() {
  rangeRlvr.addEventListener('input', (e) => {
    valRlvr.textContent = `${e.target.value}%`;
  });
  rangeGuard.addEventListener('input', (e) => {
    valGuard.textContent = `${e.target.value}%`;
  });
  rangeCuriosity.addEventListener('input', (e) => {
    valCuriosity.textContent = `${e.target.value}%`;
  });
}

function setupFormListeners() {
  const inputs = [
    inputModelName, selectDModel, selectLayers, selectHeads,
    selectExperts, selectTopK, selectSeqLen, selectVocab
  ];
  inputs.forEach(el => {
    el.addEventListener('change', () => triggerRecalculation());
  });
  inputModelName.addEventListener('input', () => debounceRecalculate());
}

function setupDomainCards() {
  const cards = document.querySelectorAll('.domain-card');
  cards.forEach(c => {
    c.addEventListener('click', () => {
      c.classList.toggle('active');
    });
  });
}

async function loadPresets() {
  try {
    const res = await fetch('/api/builder/presets');
    if (!res.ok) return;
    PRESETS = await res.json();

    presetsContainer.innerHTML = '';
    Object.keys(PRESETS).forEach((k, idx) => {
      const p = PRESETS[k];
      const card = document.createElement('div');
      card.className = `preset-card ${idx === 2 ? 'active' : ''}`;
      card.innerHTML = `
        <span class="preset-name">${p.name}</span>
        <span class="preset-sub">${p.num_experts} Exp · ${p.d_model}d</span>
      `;
      card.addEventListener('click', () => applyPreset(k, card));
      presetsContainer.appendChild(card);
    });
  } catch (err) {
    console.error('Fehler beim Laden der Presets:', err);
  }
}

function applyPreset(presetKey, cardElement) {
  const p = PRESETS[presetKey];
  if (!p) return;

  document.querySelectorAll('.preset-card').forEach(el => el.classList.remove('active'));
  if (cardElement) cardElement.classList.add('active');

  inputModelName.value = p.name;
  selectDModel.value = p.d_model;
  selectLayers.value = p.n_layers;
  selectHeads.value = p.n_heads;
  selectExperts.value = p.num_experts;
  selectTopK.value = p.top_k;
  selectSeqLen.value = p.max_seq_len;
  selectVocab.value = p.vocab_size;

  triggerRecalculation();
}

let debounceTimer = null;
function debounceRecalculate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => triggerRecalculation(), 300);
}

async function triggerRecalculation() {
  const payload = {
    name: inputModelName.value || 'Custom-MoE',
    d_model: parseInt(selectDModel.value),
    n_layers: parseInt(selectLayers.value),
    n_heads: parseInt(selectHeads.value),
    num_experts: parseInt(selectExperts.value),
    top_k: parseInt(selectTopK.value),
    max_seq_len: parseInt(selectSeqLen.value),
    vocab_size: parseInt(selectVocab.value),
    guardrails: {
      rlvr_strictness: parseInt(rangeRlvr.value) / 100,
      hallucination_guard: parseInt(rangeGuard.value) / 100,
      epistemic_curiosity: parseInt(rangeCuriosity.value) / 100,
    }
  };

  try {
    const res = await fetch('/api/builder/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) return;
    const data = await res.json();
    CURRENT_SPECS = data;
    renderCalculations(data, payload);
  } catch (err) {
    console.error('Berechnungsfehler:', err);
  }
}

function renderCalculations(data, config) {
  const params = data.params;

  // Hero Stats
  statTotalParams.textContent = `${params.total_params_billion} Mrd.`;
  statTotalSub.textContent = `${config.n_layers} Schichten · ${config.num_experts} Experten`;

  statActiveParams.textContent = `${params.active_params_million} Mio.`;
  statActiveSub.textContent = `Top-${config.top_k} Routing`;

  statSparsity.textContent = `${params.sparsity_ratio}%`;

  // Feasibility Banner
  feasibilityTitle.textContent = data.tier_badge;
  if (data.fits_rtx3060_train) {
    feasibilityIcon.textContent = '🟢';
    feasibilityBanner.style.borderColor = 'hsla(142, 76%, 45%, 0.4)';
    feasibilityBanner.style.background = 'hsla(142, 76%, 45%, 0.12)';
    feasibilityDesc.textContent = `16-Bit Training mit GaLore & JIT Layer Offloading erfordert nur ${data.training_gpu_vram_gb} GB VRAM.`;
  } else if (data.fits_rtx3060_infer_hybrid) {
    feasibilityIcon.textContent = '🟡';
    feasibilityBanner.style.borderColor = 'hsla(38, 92%, 50%, 0.4)';
    feasibilityBanner.style.background = 'hsla(38, 92%, 50%, 0.12)';
    feasibilityDesc.textContent = `Inferenz läuft als Hybrid-MoE (Top-${config.top_k}) flüssig auf 6 GB VRAM + ${data.inference_ram_hybrid_4bit_gb} GB RAM.`;
  } else {
    feasibilityIcon.textContent = '🔴';
    feasibilityBanner.style.borderColor = 'hsla(0, 84%, 60%, 0.4)';
    feasibilityBanner.style.background = 'hsla(0, 84%, 60%, 0.12)';
    feasibilityDesc.textContent = `Mega-Modell (${params.total_params_billion}B): Nutzt NVMe PCIe 4.0 Streaming für flüssige Inferenz.`;
  }

  // Memory Spec Cards
  memTrainGpu.textContent = `${data.training_gpu_vram_gb} GB`;
  memTrainRam.textContent = `${data.training_sys_ram_gb} GB`;
  memInferVram.textContent = `${data.inference_vram_4bit_full_gpu_gb} GB`;
  memInferHybridRam.textContent = `${data.inference_ram_hybrid_4bit_gb} GB`;

  // PyTorch Code
  codeOutput.textContent = data.pytorch_code || '# Generiere Code...';
}

// Copy Code Button
btnCopyCode.addEventListener('click', () => {
  const code = codeOutput.textContent;
  navigator.clipboard.writeText(code);
  showToast('📋 PyTorch Quellcode in die Zwischenablage kopiert!');
});

// Generate Model Button
btnGenerateModel.addEventListener('click', async () => {
  btnGenerateModel.disabled = true;
  btnGenerateModel.innerHTML = '<span>⏳ Kompiliere Modell...</span>';

  const payload = {
    name: inputModelName.value || 'Custom_MoE',
    d_model: parseInt(selectDModel.value),
    n_layers: parseInt(selectLayers.value),
    n_heads: parseInt(selectHeads.value),
    num_experts: parseInt(selectExperts.value),
    top_k: parseInt(selectTopK.value),
    max_seq_len: parseInt(selectSeqLen.value),
    vocab_size: parseInt(selectVocab.value),
    guardrails: {
      rlvr_strictness: parseInt(rangeRlvr.value) / 100,
      hallucination_guard: parseInt(rangeGuard.value) / 100,
      epistemic_curiosity: parseInt(rangeCuriosity.value) / 100,
    }
  };

  try {
    const res = await fetch('/api/builder/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    showToast(`🎉 ${result.message} (Gespeichert in: ${result.output_dir})`);
  } catch (err) {
    showToast(`❌ Fehler beim Generieren: ${err.message}`);
  } finally {
    btnGenerateModel.disabled = false;
    btnGenerateModel.innerHTML = '<span>⚡ Modell-Architektur Kompilieren & Speichern</span>';
  }
});

// Export Recipe Button
btnExportRecipe.addEventListener('click', () => {
  const recipe = {
    name: inputModelName.value,
    timestamp: new Date().toISOString(),
    specs: CURRENT_SPECS,
  };
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(recipe, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `${inputModelName.value.toLowerCase().replace(/[^a-z0-9]/g, '_')}_recipe.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
  showToast('💾 Trainings-Rezept (.json) erfolgreich heruntergeladen!');
});

function showToast(msg) {
  builderToast.textContent = msg;
  builderToast.style.display = 'block';
  setTimeout(() => {
    builderToast.style.display = 'none';
  }, 4500);
}
