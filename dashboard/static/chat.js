/**
 * Universal AI Chat Studio | Client Engine
 * Features: SSE Token Streaming, <think> Collapsible Reasoning, Claude Artifacts Live Sandbox,
 * Copilot Code Blocks with Syntax Copy, Chat History Persistence & Speech Recognition.
 */

// State Management
let currentPersona = 'claude'; // claude, copilot, chatgpt, gemini
let currentSessionId = null;
let chatSessions = [];
let isGenerating = false;
let activeAbortController = null;

// DOM Elements
const messagesContainer = document.getElementById('chat-messages-container');
const welcomeHero = document.getElementById('welcome-hero');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');
const btnVoice = document.getElementById('btn-voice');
const fileInput = document.getElementById('file-input');
const btnAttach = document.getElementById('btn-attach');
const toggleThinking = document.getElementById('toggle-thinking');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('chat-sidebar');
const historyContainer = document.getElementById('chat-history-container');
const newChatBtn = document.getElementById('new-chat-btn');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const chatSearchInput = document.getElementById('chat-search-input');
const canvasToggleBtn = document.getElementById('canvas-toggle-btn');
const artifactsPanel = document.getElementById('artifacts-panel');
const btnCloseArtifact = document.getElementById('btn-close-artifact');
const btnCopyArtifact = document.getElementById('btn-copy-artifact');
const btnRunArtifact = document.getElementById('btn-run-artifact');
const artifactFrame = document.getElementById('artifact-sandbox-frame');
const artifactRawCode = document.getElementById('artifact-raw-code');
const artifactCodeView = document.getElementById('artifact-code-view');

// 1. Initial Load & Session Setup
document.addEventListener('DOMContentLoaded', () => {
  loadSessionsFromStorage();
  setupPersonaButtons();
  setupEventListeners();
  autoResizeTextarea();
  
  if (chatSessions.length === 0) {
    createNewSession();
  } else {
    loadSession(chatSessions[0].id);
  }
});

// 2. Chat Session Management
function loadSessionsFromStorage() {
  try {
    const saved = localStorage.getItem('universal_ai_chat_sessions');
    chatSessions = saved ? JSON.parse(saved) : [];
  } catch (e) {
    chatSessions = [];
  }
}

function saveSessionsToStorage() {
  localStorage.setItem('universal_ai_chat_sessions', JSON.stringify(chatSessions));
  renderHistoryList();
}

function createNewSession() {
  const newId = 'session_' + Date.now();
  const newSession = {
    id: newId,
    title: 'Neuer Chat',
    persona: currentPersona,
    createdAt: new Date().toISOString(),
    messages: []
  };
  chatSessions.unshift(newSession);
  saveSessionsToStorage();
  loadSession(newId);
  if (userInput) userInput.focus();
}

function loadSession(sessionId) {
  currentSessionId = sessionId;
  const session = chatSessions.find(s => s.id === sessionId);
  if (!session) return;

  // Set persona
  if (session.persona) {
    setPersona(session.persona);
  }

  // Clear messages container
  messagesContainer.innerHTML = '';
  if (session.messages.length === 0) {
    if (welcomeHero) messagesContainer.appendChild(welcomeHero);
  } else {
    session.messages.forEach(msg => {
      renderMessageRow(msg.role, msg.content, false);
    });
  }

  renderHistoryList();
  scrollToBottom();
}

function renderHistoryList(filterQuery = '') {
  if (!historyContainer) return;
  historyContainer.innerHTML = '';

  const filtered = chatSessions.filter(s => 
    s.title.toLowerCase().includes(filterQuery.toLowerCase())
  );

  filtered.forEach(session => {
    const item = document.createElement('div');
    item.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
    item.textContent = session.title;
    item.onclick = () => loadSession(session.id);
    historyContainer.appendChild(item);
  });
}

// 3. Persona Selector
function setupPersonaButtons() {
  document.querySelectorAll('.persona-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = btn.getAttribute('data-persona');
      setPersona(p);
    });
  });
}

function setPersona(personaKey) {
  currentPersona = personaKey;
  document.querySelectorAll('.persona-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-persona') === personaKey);
  });
  const currentSession = chatSessions.find(s => s.id === currentSessionId);
  if (currentSession) {
    currentSession.persona = currentPersona;
    saveSessionsToStorage();
  }
}

// 4. Message Rendering & Markdown / Code Parser
function renderMessageRow(role, content, animate = false) {
  if (welcomeHero && welcomeHero.parentNode === messagesContainer) {
    messagesContainer.removeChild(welcomeHero);
  }

  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const personaNames = {
    reasoning: 'Deep Reasoning',
    code: 'Code Architect',
    dialog: 'Allround Dialog',
    polymath: 'Universal Polymath',
    claude: 'Deep Reasoning',
    copilot: 'Code Architect',
    chatgpt: 'Allround Dialog',
    gemini: 'Universal Polymath'
  };

  const personaIcons = {
    reasoning: '🧠',
    code: '⚡',
    dialog: '💬',
    polymath: '🌐',
    claude: '🧠',
    copilot: '⚡',
    chatgpt: '💬',
    gemini: '🌐'
  };

  const currentName = personaNames[currentPersona] || 'KI-Experte';
  const currentIcon = personaIcons[currentPersona] || '🌌';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? '👤' : currentIcon;

  const contentWrapper = document.createElement('div');
  contentWrapper.className = 'msg-content-wrapper';

  const senderLabel = document.createElement('span');
  senderLabel.className = 'msg-sender';
  senderLabel.textContent = role === 'user' ? 'Du' : `${currentName} · 7.45B MoE`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  // Parse <think> reasoning tags and Code Blocks
  bubble.innerHTML = parseRichContent(content);

  contentWrapper.appendChild(senderLabel);
  contentWrapper.appendChild(bubble);

  row.appendChild(avatar);
  row.appendChild(contentWrapper);

  messagesContainer.appendChild(row);
  scrollToBottom();
  return bubble;
}

function parseRichContent(text) {
  if (!text) return '';

  let parsed = text;

  // 1. Parse <think> ... </think> reasoning blocks
  if (parsed.includes('<think>')) {
    const parts = parsed.split('</think>');
    let thinkPart = parts[0].replace('<think>', '').trim();
    let mainPart = parts.length > 1 ? parts[1].trim() : '';

    parsed = `
      <div class="reasoning-box">
        <div class="reasoning-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
          <span>🧠 Denkprozess & Analyse</span>
          <span style="font-size: 0.7rem;">(Klicken zum Ausklappen)</span>
        </div>
        <div class="reasoning-content">${escapeHtml(thinkPart)}</div>
      </div>
      ${parseMarkdown(mainPart)}
    `;
  } else {
    parsed = parseMarkdown(parsed);
  }

  return parsed;
}

function parseMarkdown(md) {
  if (!md) return '';

  // 1. Code Blocks ```lang ... ```
  let formatted = md.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const l = lang || 'code';
    const escapedCode = escapeHtml(code.trim());
    const isWebCode = ['html', 'svg', 'xml', 'javascript', 'js', 'css'].includes(l.toLowerCase());
    
    return `
      <div class="code-block-wrapper">
        <div class="code-header">
          <span>${l.toUpperCase()}</span>
          <div class="code-actions">
            ${isWebCode ? `<button type="button" onclick="openInCanvas('${encodeURIComponent(code)}')">🎨 In Canvas ansehen</button>` : ''}
            <button type="button" onclick="copyCodeText(this, '${encodeURIComponent(code)}')">📋 Kopieren</button>
          </div>
        </div>
        <pre><code class="language-${l}">${escapedCode}</code></pre>
      </div>
    `;
  });

  // 2. Bold, Italic, Headers & Lists
  formatted = formatted
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code style="background:hsla(217,33%,25%,0.5);padding:2px 6px;border-radius:4px;font-family:var(--font-mono);">$1</code>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');

  return formatted;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

window.copyCodeText = function(btn, encodedCode) {
  const code = decodeURIComponent(encodedCode);
  navigator.clipboard.writeText(code).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✅ Kopiert!';
    setTimeout(() => btn.textContent = orig, 2000);
  });
};

window.openInCanvas = function(encodedCode) {
  const code = decodeURIComponent(encodedCode);
  if (artifactsPanel) artifactsPanel.classList.remove('collapsed');
  if (artifactRawCode) artifactRawCode.textContent = code;
  if (artifactFrame) {
    artifactFrame.srcdoc = code.includes('<html') ? code : `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{background:#0f172a;color:#f8fafc;font-family:sans-serif;padding:1.5rem;}</style></head><body>${code}</body></html>`;
  }
};

// 5. Sending & Streaming Logic (SSE)
async function handleSendMessage() {
  if (isGenerating || !userInput) return;
  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = '';
  userInput.style.height = 'auto';

  const currentSession = chatSessions.find(s => s.id === currentSessionId);
  if (!currentSession) return;

  // Add User Message
  currentSession.messages.push({ role: 'user', content: text });
  if (currentSession.title === 'Neuer Chat') {
    currentSession.title = text.slice(0, 28) + (text.length > 28 ? '...' : '');
  }
  saveSessionsToStorage();

  renderMessageRow('user', text);

  // Prepare Assistant Placeholder Bubble
  const botBubble = renderMessageRow('assistant', '⚡ Analysiere...');
  isGenerating = true;

  // Intercept /lint command for automatic folder syntax & bracket scan
  if (text.startsWith('/lint')) {
    const targetFolder = text.replace('/lint', '').trim() || '/home/benjamin/Bilder';
    try {
      const lintRes = await fetch('/api/lint_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: targetFolder })
      });
      const lintData = await lintRes.json();
      
      let reportMd = `### 🔍 Ordner-Scan & Linting Bericht\n\n`;
      reportMd += `**📁 Verzeichnis:** \`${lintData.target_dir}\`\n`;
      reportMd += `**📄 Gescannte Dateien:** ${lintData.scanned_files} (Python, JS, TS, Java, HTML, CSS, JSON)\n`;
      reportMd += `**⚠️ Gefundene Fehler:** ${lintData.error_count}\n\n`;

      if (lintData.error_count === 0) {
        reportMd += `✅ **Perfekt!** Es wurden keine unbalancierten Klammern, Syntaxfehler oder fehlerhaften JSON-Dateien gefunden. Der Code ist sauber strukturiert.`;
      } else {
        reportMd += `| Datei | Zeile:Spalte | Typ | Problem-Beschreibung |\n| :--- | :--- | :--- | :--- |\n`;
        lintData.errors.forEach(err => {
          reportMd += `| \`${err.file}\` | \`Z.${err.line}:S.${err.col}\` | **${err.type}** | ${err.msg} |\n`;
        });
        reportMd += `\n\n💡 **Empfehlung der KI:** Überprüfe die oben gelisteten Dateien an den exakten Zeilennummern.`;
      }

      botBubble.innerHTML = parseRichContent(reportMd);
      currentSession.messages.push({ role: 'assistant', content: reportMd });
      saveSessionsToStorage();
      isGenerating = false;
      return;
    } catch (e) {
      botBubble.textContent = 'Fehler beim Ausführen des Ordner-Linters: ' + e;
      isGenerating = false;
      return;
    }
  }

  try {
    const toggleWebSearch = document.getElementById('toggle-web-search');
    const isWebSearch = (toggleWebSearch && toggleWebSearch.checked) || text.startsWith('/web');

    if (isWebSearch) {
      botBubble.innerHTML = '<span style="color: var(--primary);">🌐 Durchsuche das Live-Internet nach aktuellen Informationen...</span>';
    }

    const payload = {
      messages: currentSession.messages,
      persona: currentPersona,
      thinking_enabled: toggleThinking ? toggleThinking.checked : true,
      web_search_enabled: isWebSearch,
      max_tokens: 120,
      temperature: 0.7
    };

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      botBubble.textContent = 'Fehler bei der Verbindung zum Modell-Server.';
      isGenerating = false;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let accumulatedText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
              accumulatedText += data.token;
              botBubble.innerHTML = parseRichContent(accumulatedText);
              scrollToBottom();
            }
            if (data.done && data.full_reply) {
              accumulatedText = data.full_reply;
              botBubble.innerHTML = parseRichContent(accumulatedText);
            }
          } catch (err) {
            // Ignore partial SSE json chunks
          }
        }
      }
    }

    // Save Assistant reply to session
    currentSession.messages.push({ role: 'assistant', content: accumulatedText });
    saveSessionsToStorage();

    // Check if code/html was generated and automatically populate artifacts preview
    if (accumulatedText.includes('```html') || accumulatedText.includes('```svg')) {
      const match = accumulatedText.match(/```(?:html|svg)\n([\s\S]*?)```/);
      if (match && match[1]) {
        openInCanvas(encodeURIComponent(match[1]));
      }
    }

  } catch (err) {
    botBubble.textContent = 'Netzwerkfehler oder Inferenz unterbrochen: ' + err;
  } finally {
    isGenerating = false;
  }
}

// 6. UI Helpers & Event Listeners
function setupEventListeners() {
  if (btnSend) btnSend.addEventListener('click', handleSendMessage);
  
  if (userInput) {
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
    userInput.addEventListener('input', autoResizeTextarea);
  }

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });
  }

  if (newChatBtn) newChatBtn.addEventListener('click', createNewSession);

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      if (confirm('Möchtest du wirklich alle Chat-Verläufe löschen?')) {
        chatSessions = [];
        saveSessionsToStorage();
        createNewSession();
      }
    });
  }

  if (chatSearchInput) {
    chatSearchInput.addEventListener('input', (e) => {
      renderHistoryList(e.target.value);
    });
  }

  if (canvasToggleBtn && artifactsPanel) {
    canvasToggleBtn.addEventListener('click', () => {
      artifactsPanel.classList.toggle('collapsed');
    });
  }

  if (btnCloseArtifact && artifactsPanel) {
    btnCloseArtifact.addEventListener('click', () => {
      artifactsPanel.classList.add('collapsed');
    });
  }

  // Artifact Tabs
  document.querySelectorAll('.art-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.art-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-tab');
      const memoryView = document.getElementById('artifact-memory-view');

      if (target === 'preview') {
        if (artifactFrame) artifactFrame.style.display = 'block';
        if (artifactCodeView) artifactCodeView.style.display = 'none';
        if (memoryView) memoryView.style.display = 'none';
      } else if (target === 'code') {
        if (artifactFrame) artifactFrame.style.display = 'none';
        if (artifactCodeView) artifactCodeView.style.display = 'block';
        if (memoryView) memoryView.style.display = 'none';
      } else if (target === 'memory') {
        if (artifactFrame) artifactFrame.style.display = 'none';
        if (artifactCodeView) artifactCodeView.style.display = 'none';
        if (memoryView) {
          memoryView.style.display = 'block';
          loadMemoryGraph();
        }
      }
    });
  });

  // Top Open Memory Button
  const btnOpenMemory = document.getElementById('btn-open-memory');
  if (btnOpenMemory && artifactsPanel) {
    btnOpenMemory.addEventListener('click', () => {
      artifactsPanel.classList.remove('collapsed');
      const memTab = document.querySelector('.art-tab[data-tab="memory"]');
      if (memTab) memTab.click();
    });
  }

  // Refresh Memory Button
  const btnRefreshMem = document.getElementById('btn-refresh-memory');
  if (btnRefreshMem) {
    btnRefreshMem.addEventListener('click', loadMemoryGraph);
  }

  // Save Fact Button
  const btnSaveFact = document.getElementById('btn-save-fact');
  if (btnSaveFact) {
    btnSaveFact.addEventListener('click', async () => {
      const src = document.getElementById('mem-source')?.value.trim();
      const rel = document.getElementById('mem-relation')?.value.trim();
      const tgt = document.getElementById('mem-target')?.value.trim();
      if (!src || !rel || !tgt) {
        alert('Bitte alle 3 Felder ausfüllen (Subjekt, Beziehung, Objekt).');
        return;
      }
      try {
        const res = await fetch('/api/memory/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: src, relation: rel, target: tgt })
        });
        const data = await res.json();
        if (data.status === 'ok') {
          document.getElementById('mem-source').value = '';
          document.getElementById('mem-relation').value = '';
          document.getElementById('mem-target').value = '';
          loadMemoryGraph();
        }
      } catch (err) {
        alert('Fehler beim Speichern: ' + err);
      }
    });
  }

  // Voice Input (Web Speech API)
  if (btnVoice) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'de-DE';
      recognition.continuous = false;

      recognition.onstart = () => {
        btnVoice.style.color = '#ef4444';
      };
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (userInput) {
          userInput.value += (userInput.value ? ' ' : '') + transcript;
          autoResizeTextarea();
        }
      };
      recognition.onend = () => {
        btnVoice.style.color = '';
      };

      btnVoice.addEventListener('click', () => {
        try { recognition.start(); } catch (e) {}
      });
    } else {
      btnVoice.style.display = 'none';
    }
  }

  // File Upload
  if (btnAttach && fileInput) {
    btnAttach.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (userInput) {
          userInput.value += `\n[Angehängte Datei: ${file.name}]\n${ev.target.result}\n`;
          autoResizeTextarea();
        }
      };
      reader.readAsText(file);
    });
  }

  // Keyboard Shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
      e.preventDefault();
      createNewSession();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
      e.preventDefault();
      if (sidebar) sidebar.classList.toggle('collapsed');
    }
  });
}

function autoResizeTextarea() {
  if (!userInput) return;
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
}

function scrollToBottom() {
  if (messagesContainer) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

window.setQuickPrompt = function(promptText) {
  if (userInput) {
    userInput.value = promptText;
    autoResizeTextarea();
    handleSendMessage();
  }
};

window.insertSlashCommand = function(cmd) {
  if (userInput) {
    userInput.value = cmd + userInput.value;
    userInput.focus();
    autoResizeTextarea();
  }
};

async function loadMemoryGraph() {
  const statsEl = document.getElementById('memory-graph-stats');
  const listEl = document.getElementById('memory-triplets-list');
  if (!statsEl || !listEl) return;

  statsEl.textContent = 'Lade Bitstream-Knoten...';
  listEl.innerHTML = '';

  try {
    const res = await fetch('/api/memory/graph');
    const data = await res.json();

    statsEl.innerHTML = `📊 <strong>${data.total_nodes} Entitäten</strong> · <strong>${data.total_links} Relationen</strong> im 20-Bit Golden Master Bitstream`;

    if (!data.links || data.links.length === 0) {
      listEl.innerHTML = '<div style="font-size:0.75rem;color:var(--text-muted);">Keine Fakten gespeichert.</div>';
      return;
    }

    const catColors = {
      user: '#38bdf8',
      project: '#a855f7',
      tech: '#22c55e',
      minecraft: '#eab308',
      math: '#f97316',
      system: '#ec4899',
      concept: '#94a3b8'
    };

    data.links.forEach(l => {
      const srcNode = data.nodes.find(n => n.id === l.source) || { label: l.source, category: 'concept' };
      const tgtNode = data.nodes.find(n => n.id === l.target) || { label: l.target, category: 'concept' };

      const card = document.createElement('div');
      card.style.cssText = 'background: hsla(222,47%,14%,0.7); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 6px 10px; font-size: 0.78rem; display: flex; align-items: center; justify-content: space-between;';
      
      const srcColor = catColors[srcNode.category] || '#38bdf8';
      const tgtColor = catColors[tgtNode.category] || '#a855f7';

      card.innerHTML = `
        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
          <span style="color: ${srcColor}; font-weight: 600;">${escapeHtml(srcNode.label)}</span>
          <span style="color: var(--text-muted); font-size: 0.7rem; font-family: var(--font-mono); background: hsla(217,33%,25%,0.5); padding: 1px 5px; border-radius: 4px;">--[${escapeHtml(l.relation)}]--></span>
          <span style="color: ${tgtColor}; font-weight: 600;">${escapeHtml(tgtNode.label)}</span>
        </div>
      `;
      listEl.appendChild(card);
    });

  } catch (err) {
    statsEl.textContent = 'Fehler beim Laden des Wissensgraphen: ' + err;
  }
}
