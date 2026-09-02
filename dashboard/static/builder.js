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

const selectHardwareProfile = document.getElementById('select-hardware-profile');
const selectQuantization = document.getElementById('select-quantization');
const btnModeDense = document.getElementById('btn-mode-dense');
const btnModeMoe = document.getElementById('btn-mode-moe');
const groupTopK = document.getElementById('group-top-k');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  setupSliderListeners();
  setupModeSwitcher();
  setupFormListeners();
  setupDomainCards();
  await loadPresets();
  triggerRecalculation();
});

function setupModeSwitcher() {
  if (btnModeDense) {
    btnModeDense.addEventListener('click', () => {
      btnModeDense.classList.add('active');
      btnModeMoe.classList.remove('active');
      selectExperts.value = '1';
      selectTopK.value = '1';
      if (groupTopK) groupTopK.style.opacity = '0.5';
      triggerRecalculation();
    });
  }

  if (btnModeMoe) {
    btnModeMoe.addEventListener('click', () => {
      btnModeMoe.classList.add('active');
      btnModeDense.classList.remove('active');
      if (selectExperts.value === '1') {
        selectExperts.value = '12';
        selectTopK.value = '2';
      }
      if (groupTopK) groupTopK.style.opacity = '1.0';
      triggerRecalculation();
    });
  }

  const multButtons = document.querySelectorAll('.btn-mult');
  multButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      multButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = btn.dataset.mult;
      selectExperts.value = val;
      if (val === '1') {
        if (btnModeDense) btnModeDense.classList.add('active');
        if (btnModeMoe) btnModeMoe.classList.remove('active');
        if (groupTopK) groupTopK.style.opacity = '0.5';
      } else {
        if (btnModeMoe) btnModeMoe.classList.add('active');
        if (btnModeDense) btnModeDense.classList.remove('active');
        if (groupTopK) groupTopK.style.opacity = '1.0';
      }
      triggerRecalculation();
    });
  });

  if (selectExperts) {
    selectExperts.addEventListener('change', () => {
      const val = selectExperts.value;
      multButtons.forEach(b => {
        if (b.dataset.mult === val) b.classList.add('active');
        else b.classList.remove('active');
      });

      if (val === '1') {
        if (btnModeDense) btnModeDense.classList.add('active');
        if (btnModeMoe) btnModeMoe.classList.remove('active');
        if (groupTopK) groupTopK.style.opacity = '0.5';
      } else {
        if (btnModeMoe) btnModeMoe.classList.add('active');
        if (btnModeDense) btnModeDense.classList.remove('active');
        if (groupTopK) groupTopK.style.opacity = '1.0';
      }
    });
  }
}

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
    selectExperts, selectTopK, selectSeqLen, selectVocab,
    selectHardwareProfile, selectQuantization
  ];
  inputs.forEach(el => {
    if (el) el.addEventListener('change', () => triggerRecalculation());
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

  // Sync multiplier buttons
  const multButtons = document.querySelectorAll('.btn-mult');
  multButtons.forEach(b => {
    if (b.dataset.mult == p.num_experts) b.classList.add('active');
    else b.classList.remove('active');
  });

  // Sync mode buttons
  if (p.num_experts === 1) {
    if (btnModeDense) btnModeDense.classList.add('active');
    if (btnModeMoe) btnModeMoe.classList.remove('active');
    if (groupTopK) groupTopK.style.opacity = '0.5';
  } else {
    if (btnModeMoe) btnModeMoe.classList.add('active');
    if (btnModeDense) btnModeDense.classList.remove('active');
    if (groupTopK) groupTopK.style.opacity = '1.0';
  }

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

// Download Complete Training Bundle (.zip)
const btnDownloadBundle = document.getElementById('btn-download-bundle');
if (btnDownloadBundle) {
  btnDownloadBundle.addEventListener('click', async () => {
    btnDownloadBundle.disabled = true;
    btnDownloadBundle.innerHTML = '<span>⏳ Schnüre Paket mit Vokabular...</span>';

    const rawName = (inputModelName && inputModelName.value) ? inputModelName.value.trim() : 'Custom_MoE';
    const modelName = rawName.replace(/\s+/g, '_') || 'Custom_MoE';

    const payload = {
      name: modelName,
      d_model: parseInt(selectDModel.value) || 2048,
      n_layers: parseInt(selectLayers.value) || 24,
      n_heads: parseInt(selectHeads.value) || 16,
      num_experts: parseInt(selectExperts ? selectExperts.value : 12) || 12,
      top_k: parseInt(selectTopK ? selectTopK.value : 2) || 2,
      max_seq_len: parseInt(selectSeqLen.value) || 7168,
      vocab_size: parseInt(selectVocab ? selectVocab.value : 1048576) || 1048576,
      guardrails: {
        rlvr_strictness: rangeRlvr ? parseInt(rangeRlvr.value) / 100 : 0.85,
        hallucination_guard: rangeGuard ? parseInt(rangeGuard.value) / 100 : 0.90,
        epistemic_curiosity: rangeCuriosity ? parseInt(rangeCuriosity.value) / 100 : 0.75,
      }
    };

    try {
      const qParams = new URLSearchParams({
        name: payload.name,
        d_model: payload.d_model,
        n_layers: payload.n_layers,
        n_heads: payload.n_heads,
        num_experts: payload.num_experts,
        top_k: payload.top_k,
        max_seq_len: payload.max_seq_len,
        vocab_size: payload.vocab_size,
        rlvr_strictness: payload.guardrails.rlvr_strictness,
        hallucination_guard: payload.guardrails.hallucination_guard,
        epistemic_curiosity: payload.guardrails.epistemic_curiosity
      });
      const downloadUrl = `/api/builder/download_bundle?${qParams.toString()}`;

      const downloadAnchor = document.createElement('a');
      downloadAnchor.href = downloadUrl;
      downloadAnchor.setAttribute('download', `${payload.name}_training_bundle.zip`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      setTimeout(() => downloadAnchor.remove(), 1000);

      showToast(`📦 Trainings-Paket '${payload.name}_training_bundle.zip' (inkl. Vokabular) wird heruntergeladen!`);
    } catch (err) {
      showToast(`❌ Fehler beim Herunterladen: ${err.message}`);
    } finally {
      setTimeout(() => {
        btnDownloadBundle.disabled = false;
        btnDownloadBundle.innerHTML = '<span>📦 Komplettes Trainings-Paket (.zip) Herunterladen</span>';
      }, 2000);
    }
  });
}

function showToast(msg) {
  builderToast.textContent = msg;
  builderToast.style.display = 'block';
  setTimeout(() => {
    builderToast.style.display = 'none';
  }, 4500);
}
