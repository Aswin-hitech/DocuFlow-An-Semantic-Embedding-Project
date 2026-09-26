document.addEventListener('DOMContentLoaded', () => {
  // Base API configuration
  const API_BASE = window.location.origin;

  // Conversational State
  let chatHistory = [];
  let selectedFile = null;
  let activeSourcesList = [];

  // DOM Elements - Status & Header
  const backendStatus = document.getElementById('backend-status');
  const backendStatusText = document.getElementById('backend-status-text');
  const modelPill = document.getElementById('model-pill');
  const chunkCountPill = document.getElementById('chunk-count-pill');
  const footerLlmLabel = document.getElementById('footer-llm-label');
  const toast = document.getElementById('toast');

  // DOM Elements - Navigation Tabs (Apple Segmented Control)
  const tabs = document.querySelectorAll('.nav-segment-btn, .tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // DOM Elements - Chatbot
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatSendBtn = document.getElementById('chat-send-btn');
  const chatMessages = document.getElementById('chat-messages');
  const chatHero = document.getElementById('chat-hero');
  const chatSourceSelect = document.getElementById('chat-source-select');
  const chatSourceBadge = document.getElementById('chat-source-badge');
  const refreshSourcesBtn = document.getElementById('refresh-sources-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const chatTopK = document.getElementById('chat-top-k');
  const chatTopKVal = document.getElementById('chat-top-k-val');
  const activeScopeBanner = document.getElementById('active-scope-banner');
  const activeScopeType = document.getElementById('active-scope-type');
  const activeScopeTitle = document.getElementById('active-scope-title');
  const activeScopeChunks = document.getElementById('active-scope-chunks');
  const resetScopeBtn = document.getElementById('reset-scope-btn');

  // DOM Elements - Ingestion Banners
  const ingestActionBanner = document.getElementById('ingest-action-banner');
  const successBannerTitle = document.getElementById('success-banner-title');
  const successBannerDesc = document.getElementById('success-banner-desc');
  const startChatWithIngestedBtn = document.getElementById('start-chat-with-ingested-btn');
  let lastIngestedSourceId = null;

  // DOM Elements - Modal
  const chunkModal = document.getElementById('chunk-modal');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalTitle = document.getElementById('modal-title');
  const modalSource = document.getElementById('modal-source');
  const modalScore = document.getElementById('modal-score');
  const modalText = document.getElementById('modal-text');
  const modalCopyBtn = document.getElementById('modal-copy-btn');
  const modalChatBtn = document.getElementById('modal-chat-btn');
  let currentModalSource = null;

  // Helper: HTML Escaping
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Helper: Apple Toast Notifications
  function showToast(message, type = 'info') {
    if (!toast) return;
    toast.textContent = message;
    toast.className = `apple-toast-pill show ${type}`;
    setTimeout(() => {
      toast.className = 'apple-toast-pill';
    }, 3200);
  }

  // Helper: Format Markdown (using marked or fallback)
  function renderMarkdown(text) {
    if (!text) return '';
    if (window.marked && typeof window.marked.parse === 'function') {
      try {
        return window.marked.parse(text);
      } catch (e) {
        console.warn('Marked parsing error:', e);
      }
    }
    // Fallback basic parser
    let parsed = escapeHtml(text);
    parsed = parsed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    parsed = parsed.replace(/\*(.*?)\*/g, '<em>$1</em>');
    parsed = parsed.replace(/`([^`]+)`/g, '<code>$1</code>');
    parsed = parsed.replace(/\n\n/g, '</p><p>');
    parsed = parsed.replace(/\n/g, '<br>');
    return `<p>${parsed}</p>`;
  }

  // --- TAB NAVIGATION (APPLE STYLE) ---
  function switchTab(tabId) {
    tabs.forEach(t => {
      if (t.getAttribute('data-tab') === tabId) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });

    tabContents.forEach(c => {
      if (c.id === tabId) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    if (tabId === 'explorer-tab') {
      loadKnowledgeBase();
    } else if (tabId === 'chat-tab') {
      setTimeout(() => chatInput && chatInput.focus(), 150);
    }
  }

  // Expose globally so hero steps can call switchTab()
  window.switchTab = switchTab;

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  // --- SLIDER BINDINGS ---
  const bindSlider = (slider, display, suffix = '') => {
    if (slider && display) {
      slider.addEventListener('input', () => {
        display.textContent = slider.value + suffix;
      });
    }
  };

  bindSlider(chatTopK, chatTopKVal);
  const topKSlider = document.getElementById('top-k-slider');
  const topKVal = document.getElementById('top-k-val');
  const minScoreSlider = document.getElementById('min-score-slider');
  const minScoreVal = document.getElementById('min-score-val');
  const fileChunkSize = document.getElementById('file-chunk-size');
  const fileChunkSizeVal = document.getElementById('file-chunk-size-val');
  const fileOverlap = document.getElementById('file-overlap');
  const fileOverlapVal = document.getElementById('file-overlap-val');
  const urlChunkSize = document.getElementById('url-chunk-size');
  const urlChunkSizeVal = document.getElementById('url-chunk-size-val');
  const urlOverlap = document.getElementById('url-overlap');
  const urlOverlapVal = document.getElementById('url-overlap-val');
  const textChunkSize = document.getElementById('text-chunk-size');
  const textChunkSizeVal = document.getElementById('text-chunk-size-val');

  bindSlider(topKSlider, topKVal);
  bindSlider(minScoreSlider, minScoreVal);
  bindSlider(fileChunkSize, fileChunkSizeVal);
  bindSlider(fileOverlap, fileOverlapVal);
  bindSlider(urlChunkSize, urlChunkSizeVal);
  bindSlider(urlOverlap, urlOverlapVal);
  bindSlider(textChunkSize, textChunkSizeVal);

  // --- SYSTEM STATUS ---
  async function checkStatus() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        backendStatus.classList.add('online');
        backendStatusText.textContent = 'Ready to help';
        chunkCountPill.textContent = data.total_indexed_chunks || 0;
      }
    } catch (err) {
      try {
        const res2 = await fetch(`${API_BASE}/api/`);
        if (res2.ok) {
          const data2 = await res2.json();
          backendStatus.classList.add('online');
          backendStatusText.textContent = 'Ready to help';
          chunkCountPill.textContent = data2.total_indexed_chunks || 0;
        }
      } catch (e) {
        backendStatus.classList.remove('online');
        backendStatusText.textContent = 'Connecting...';
      }
    }
  }

  // --- FETCH & POPULATE SOURCES (DOCUMENTS & SCRAPED WEBSITES) ---
  async function loadSources() {
    try {
      const res = await fetch(`${API_BASE}/api/sources`);
      if (!res.ok) return;
      const data = await res.json();
      activeSourcesList = data.sources || [];

      // Update Chat Source Dropdown
      const currentVal = chatSourceSelect.value;
      chatSourceSelect.innerHTML = `<option value="all">🌟 Search across all my papers &amp; websites</option>`;

      const searchFilter = document.getElementById('search-source-filter');
      if (searchFilter) {
        searchFilter.innerHTML = `<option value="all">All Documents &amp; Websites</option>`;
      }

      activeSourcesList.forEach(src => {
        const icon = src.type === 'web' ? '🌐' : (src.type === 'document' ? '📄' : '📝');
        const opt = document.createElement('option');
        opt.value = src.id;
        opt.textContent = `${icon} ${src.title} (${src.chunks_count} paragraphs)`;
        chatSourceSelect.appendChild(opt);

        if (searchFilter) {
          const opt2 = document.createElement('option');
          opt2.value = src.id;
          opt2.textContent = `${icon} ${src.title}`;
          searchFilter.appendChild(opt2);
        }
      });

      // Restore value if still present
      if (currentVal && Array.from(chatSourceSelect.options).some(o => o.value === currentVal)) {
        chatSourceSelect.value = currentVal;
      }
      updateActiveScopeBanner();
      renderSourcesSummary();
    } catch (err) {
      console.warn('Failed to load sources list:', err);
    }
  }

  function updateActiveScopeBanner() {
    const selected = chatSourceSelect.value;
    if (!selected || selected === 'all') {
      activeScopeBanner.style.display = 'none';
      chatSourceBadge.textContent = 'All';
      return;
    }

    const srcInfo = activeSourcesList.find(s => s.id === selected);
    activeScopeBanner.style.display = 'flex';
    if (srcInfo) {
      activeScopeType.textContent = srcInfo.type === 'web' ? 'Website' : 'Document';
      activeScopeTitle.textContent = srcInfo.title;
      activeScopeChunks.textContent = `(${srcInfo.chunks_count} paragraphs)`;
      chatSourceBadge.textContent = srcInfo.type === 'web' ? 'Web' : 'Paper';
    } else {
      activeScopeType.textContent = 'Selected';
      activeScopeTitle.textContent = selected;
      activeScopeChunks.textContent = '';
      chatSourceBadge.textContent = 'Filtered';
    }
  }

  if (chatSourceSelect) {
    chatSourceSelect.addEventListener('change', updateActiveScopeBanner);
  }

  if (resetScopeBtn) {
    resetScopeBtn.addEventListener('click', () => {
      chatSourceSelect.value = 'all';
      updateActiveScopeBanner();
      showToast('Now searching across all saved files', 'info');
    });
  }

  if (refreshSourcesBtn) {
    refreshSourcesBtn.addEventListener('click', async () => {
      await loadSources();
      await checkStatus();
      showToast('Updated your saved files list', 'info');
    });
  }

  // --- CHATBOT LOGIC ---
  if (chatInput) {
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  }

  // Clickable prompt suggestion chips
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt && chatInput) {
        chatInput.value = prompt;
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Clear Chat History
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      chatHistory = [];
      chatMessages.innerHTML = '';
      if (chatHero) {
        chatMessages.appendChild(chatHero);
        chatHero.style.display = 'flex';
      }
      showToast('Started fresh conversation', 'info');
    });
  }

  // Submit Chat Question
  if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = chatInput.value.trim();
      if (!question) return;

      const selectedSource = chatSourceSelect.value;
      const topK = parseInt(chatTopK.value, 10) || 3;

      // Hide hero on first question
      if (chatHero && chatHero.style.display !== 'none') {
        chatHero.style.display = 'none';
      }

      // 1. Render User Message
      appendUserMessage(question);
      chatInput.value = '';
      chatInput.style.height = 'auto';
      chatSendBtn.disabled = true;

      // 2. Show Typing Indicator
      const typingRow = appendTypingIndicator();
      scrollToBottom();

      try {
        const payload = {
          question: question,
          top_k: topK,
          source: selectedSource !== 'all' ? selectedSource : null,
          history: chatHistory
        };

        const res = await fetch(`${API_BASE}/api/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error (${res.status})`);
        }

        const data = await res.json();

        // Remove typing indicator
        typingRow.remove();

        if (footerLlmLabel && data.model) {
          footerLlmLabel.textContent = `Answered by ${data.model}`;
        }

        // 3. Render Assistant Response
        appendAssistantMessage(data.answer, data.sources || [], data.model);

        // Update history
        chatHistory.push({ role: 'user', content: question });
        chatHistory.push({ role: 'assistant', content: data.answer });

        if (chatHistory.length > 20) {
          chatHistory = chatHistory.slice(-20);
        }
      } catch (err) {
        typingRow.remove();
        appendErrorMessage(`Could not get answer: ${err.message}. Please check if the server is running.`);
        showToast(`Error: ${err.message}`, 'error');
      } finally {
        chatSendBtn.disabled = false;
        scrollToBottom();
      }
    });
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'chat-row user';
    row.innerHTML = `
      <div class="chat-avatar">You</div>
      <div class="chat-bubble-container">
        <div class="chat-bubble">${escapeHtml(text)}</div>
        <div class="chat-meta-bar" style="justify-content: flex-end;">
          <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
  }

  function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'chat-row assistant typing-indicator-row';
    row.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="typing-bubble">
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
        <span class="typing-text">Reading your papers and writing a simple answer...</span>
      </div>
    `;
    chatMessages.appendChild(row);
    return row;
  }

  function appendAssistantMessage(answer, sources, modelName) {
    const row = document.createElement('div');
    row.className = 'chat-row assistant';

    const renderedHtml = renderMarkdown(answer);
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Build Citations (Simple Village-Friendly Language)
    let citationsHtml = '';
    if (sources && sources.length > 0) {
      const pills = sources.map((src, i) => {
        const srcName = src.metadata?.source || src.metadata?.doc_id || src.metadata?.original_filename || 'Document';
        const scoreVal = src.score !== undefined ? src.score : 0;
        
        let matchLabel = 'Good match';
        if (scoreVal >= 0.7) matchLabel = 'Strong match';
        else if (scoreVal < 0.45) matchLabel = 'Fair match';

        const shortName = srcName.length > 28 ? srcName.slice(0, 25) + '...' : srcName;
        return `
          <button type="button" class="citation-pill" data-chunk-index="${i}" title="Click to view exact paragraph">
            <span>📖 Source: ${escapeHtml(shortName)}</span>
            <span class="citation-score">(${matchLabel})</span>
          </button>
        `;
      }).join('');

      citationsHtml = `
        <div class="chat-citations">
          <div class="citations-label">
            <span>Exact paragraphs read to answer this question:</span>
          </div>
          <div class="citation-pills-row">${pills}</div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble-container" style="max-width: 100%;">
        <div class="chat-bubble">
          ${renderedHtml}
          ${citationsHtml}
        </div>
        <div class="chat-meta-bar">
          <span>${timeStr}</span>
          <div class="chat-actions-row">
            <button type="button" class="action-btn-sm copy-btn" title="Copy this text">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              Copy Answer
            </button>
          </div>
        </div>
      </div>
    `;

    // Bind Copy Button
    const copyBtn = row.querySelector('.copy-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(answer).then(() => {
          showToast('Answer copied to clipboard!', 'info');
        });
      });
    }

    // Bind Citation Pill click to open preview modal
    row.querySelectorAll('.citation-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        const idx = parseInt(pill.getAttribute('data-chunk-index'), 10);
        if (sources[idx]) {
          openChunkModal(sources[idx]);
        }
      });
    });

    chatMessages.appendChild(row);
  }

  function appendErrorMessage(text) {
    const row = document.createElement('div');
    row.className = 'chat-row assistant';
    row.innerHTML = `
      <div class="chat-avatar" style="background: #ff3b30;">!</div>
      <div class="chat-bubble-container">
        <div class="chat-bubble" style="background: #fef2f2; border-color: #fecaca; color: #b91c1c;">
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // --- MODAL PREVIEW ---
  function openChunkModal(chunkData) {
    const text = chunkData.chunk || chunkData.text || '';
    const src = chunkData.metadata?.source || chunkData.metadata?.original_filename || chunkData.metadata?.doc_id || 'Document';
    const scoreVal = chunkData.score !== undefined ? `${Math.round(chunkData.score * 100)}% relevance` : 'Verified';

    modalTitle.textContent = `Original Paragraph Passage`;
    modalSource.textContent = src;
    modalScore.textContent = scoreVal;
    modalText.textContent = text;
    currentModalSource = src;

    chunkModal.classList.add('show');
  }

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener('click', () => chunkModal.classList.remove('show'));
  }
  if (chunkModal) {
    chunkModal.addEventListener('click', (e) => {
      if (e.target === chunkModal) chunkModal.classList.remove('show');
    });
  }
  if (modalCopyBtn) {
    modalCopyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(modalText.textContent).then(() => {
        showToast('Paragraph text copied', 'info');
      });
    });
  }
  if (modalChatBtn) {
    modalChatBtn.addEventListener('click', () => {
      chunkModal.classList.remove('show');
      startChatWithSource(currentModalSource);
    });
  }

  // --- QUICK SWITCH TO CHAT WITH A SPECIFIC SOURCE ---
  function startChatWithSource(sourceId) {
    if (!sourceId) return;
    switchTab('chat-tab');

    // Check if source exists in select options
    let found = false;
    for (let opt of chatSourceSelect.options) {
      if (opt.value === sourceId || opt.value.includes(sourceId) || sourceId.includes(opt.value)) {
        chatSourceSelect.value = opt.value;
        found = true;
        break;
      }
    }

    if (!found) {
      const newOpt = document.createElement('option');
      newOpt.value = sourceId;
      newOpt.textContent = `📄 ${sourceId}`;
      chatSourceSelect.appendChild(newOpt);
      chatSourceSelect.value = sourceId;
    }

    updateActiveScopeBanner();
    showToast(`Ready to ask questions about "${sourceId}"`, 'info');

    if (chatInput) {
      chatInput.placeholder = `What would you like to know about "${sourceId}"?`;
      chatInput.focus();
    }
  }

  // --- TAB 3: SEARCH WORDS ---
  const searchForm = document.getElementById('search-form');
  const searchQuery = document.getElementById('search-query');
  const searchResultsList = document.getElementById('search-results-list');
  const resultsMeta = document.getElementById('results-meta');
  const resultsCountText = document.getElementById('results-count-text');
  const searchBtn = document.getElementById('search-btn');
  const searchSourceFilter = document.getElementById('search-source-filter');

  document.querySelectorAll('.chip-apple').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (searchQuery && q) {
        searchQuery.value = q;
        executeSearch(q);
      }
    });
  });

  if (searchForm) {
    searchForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const q = searchQuery.value.trim();
      if (q) executeSearch(q);
    });
  }

  async function executeSearch(query) {
    if (!query) return;
    const topK = parseInt(topKSlider.value, 10) || 3;
    const selectedSource = searchSourceFilter ? searchSourceFilter.value : 'all';

    searchBtn.disabled = true;
    searchBtn.textContent = 'Searching...';

    try {
      const payload = {
        query,
        top_k: topK,
        source: selectedSource !== 'all' ? selectedSource : null
      };

      const res = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Search query failed');
      }

      const data = await res.json();
      renderSearchResults(data.results || []);
    } catch (err) {
      showToast(`Search error: ${err.message}`, 'error');
    } finally {
      searchBtn.disabled = false;
      searchBtn.textContent = 'Find Matches';
    }
  }

  function renderSearchResults(results) {
    resultsMeta.style.display = 'flex';
    resultsCountText.textContent = `Found ${results.length} matching paragraph${results.length !== 1 ? 's' : ''}`;
    searchResultsList.innerHTML = '';

    if (!results || results.length === 0) {
      searchResultsList.innerHTML = `
        <div class="empty-state-simple">
          <p>No matching lines found. Try using simpler words or adding more documents.</p>
        </div>
      `;
      return;
    }

    results.forEach((item, index) => {
      const score = item.score !== undefined ? item.score : 0;
      const scorePct = Math.max(0, Math.min(100, Math.round(score * 100)));

      let scoreClass = 'low';
      let scoreLabel = 'Fair Match';
      let barColor = '#f59e0b';

      if (score >= 0.6) {
        scoreClass = 'high';
        scoreLabel = '⭐⭐⭐ Strong Match';
        barColor = '#10b981';
      } else if (score >= 0.4) {
        scoreClass = 'medium';
        scoreLabel = '⭐⭐ Good Match';
        barColor = '#0071e3';
      }

      const card = document.createElement('div');
      card.className = 'result-card';
      const text = item.chunk || item.text || '';
      const source = item.metadata?.source || item.metadata?.original_filename || item.metadata?.doc_id || 'Document';

      card.innerHTML = `
        <div class="result-meta-bar">
          <span class="rank-badge">Match #${index + 1}</span>
          <span class="score-badge ${scoreClass}">${scoreLabel} (${scorePct}%)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${scorePct}%; background-color: ${barColor};"></div>
        </div>
        <div class="result-body">${escapeHtml(text)}</div>
        <div class="result-footer">
          <span>Found in: <strong>${escapeHtml(source)}</strong></span>
          <button type="button" class="chat-now-btn">💬 Ask Questions About This</button>
        </div>
      `;

      card.querySelector('.chat-now-btn').addEventListener('click', () => {
        startChatWithSource(source);
      });

      searchResultsList.appendChild(card);
    });
  }

  // --- TAB 2: ADD PAPERS & LINKS ---
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const selectedFileInfo = document.getElementById('selected-file-name');
  const uploadBtn = document.getElementById('upload-btn');
  const fileUploadForm = document.getElementById('file-upload-form');

  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    selectedFile = file;
    selectedFileInfo.style.display = 'flex';
    selectedFileInfo.innerHTML = `
      <span>📄 <strong>${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(1)} KB)</span>
      <button type="button" id="remove-file-btn" style="background:transparent;border:none;color:#ff3b30;cursor:pointer;font-weight:700;">✕ Remove</button>
    `;
    uploadBtn.disabled = false;

    document.getElementById('remove-file-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      selectedFile = null;
      fileInput.value = '';
      selectedFileInfo.style.display = 'none';
      uploadBtn.disabled = true;
    });
  }

  if (fileUploadForm) {
    fileUploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!selectedFile) return;

      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Reading & Saving...';

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('chunk_size', fileChunkSize.value);
      formData.append('overlap', fileOverlap.value);

      try {
        const res = await fetch(`${API_BASE}/api/ingest/file`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'File reading failed');
        }

        const data = await res.json();
        showToast('Document saved successfully!', 'info');

        lastIngestedSourceId = data.filename || selectedFile.name;
        showIngestBanner(
          `Document "${lastIngestedSourceId}" is Ready!`,
          `We have read the document and saved ${data.chunks_indexed} paragraphs ready for your questions.`
        );

        selectedFile = null;
        fileInput.value = '';
        selectedFileInfo.style.display = 'none';

        await loadSources();
        await checkStatus();
      } catch (err) {
        showToast(`Could not read file: ${err.message}`, 'error');
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Save & Read Document';
      }
    });
  }

  // Web Scraper Form
  const urlIngestForm = document.getElementById('url-ingest-form');
  const scrapeUrl = document.getElementById('scrape-url');
  const scrapeBtn = document.getElementById('scrape-btn');

  if (urlIngestForm) {
    urlIngestForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const url = scrapeUrl.value.trim();
      if (!url) return;

      scrapeBtn.disabled = true;
      scrapeBtn.textContent = 'Reading Website...';

      try {
        const res = await fetch(`${API_BASE}/api/ingest/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url,
            chunk_size: parseInt(urlChunkSize.value, 10),
            overlap: parseInt(urlOverlap.value, 10)
          })
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Website reading failed');
        }

        const data = await res.json();
        showToast('Website read successfully!', 'info');

        lastIngestedSourceId = data.url || url;
        showIngestBanner(
          `Website "${data.title || url}" is Ready!`,
          `Extracted the web text and saved ${data.chunks_indexed} paragraphs for quick answers.`
        );

        scrapeUrl.value = '';
        await loadSources();
        await checkStatus();
      } catch (err) {
        showToast(`Could not read website: ${err.message}`, 'error');
      } finally {
        scrapeBtn.disabled = false;
        scrapeBtn.textContent = 'Read Website Link';
      }
    });
  }

  // Raw Text Ingestion Form
  const textIngestForm = document.getElementById('text-ingest-form');
  const textTitle = document.getElementById('text-title');
  const rawTextContent = document.getElementById('raw-text-content');
  const textBtn = document.getElementById('text-btn');

  if (textIngestForm) {
    textIngestForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const title = textTitle.value.trim();
      const text = rawTextContent.value.trim();
      if (!title || !text) return;

      textBtn.disabled = true;
      textBtn.textContent = 'Saving Notes...';

      try {
        const res = await fetch(`${API_BASE}/api/ingest/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: title,
            text: text,
            chunk_size: parseInt(textChunkSize.value, 10)
          })
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Text saving failed');
        }

        const data = await res.json();
        showToast('Notes saved successfully!', 'info');

        lastIngestedSourceId = title;
        showIngestBanner(
          `Notes "${title}" are Ready!`,
          `Saved ${data.chunks_indexed} paragraphs ready for questions.`
        );

        textTitle.value = '';
        rawTextContent.value = '';
        await loadSources();
        await checkStatus();
      } catch (err) {
        showToast(`Could not save notes: ${err.message}`, 'error');
      } finally {
        textBtn.disabled = false;
        textBtn.textContent = 'Save These Notes';
      }
    });
  }

  function showIngestBanner(title, desc) {
    if (!ingestActionBanner) return;
    successBannerTitle.textContent = title;
    successBannerDesc.textContent = desc;
    ingestActionBanner.style.display = 'flex';
    ingestActionBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  if (startChatWithIngestedBtn) {
    startChatWithIngestedBtn.addEventListener('click', () => {
      if (lastIngestedSourceId) {
        startChatWithSource(lastIngestedSourceId);
      } else {
        switchTab('chat-tab');
      }
    });
  }

  // --- TAB 4: MY SAVED FILES (KNOWLEDGE BASE) ---
  const expTotalChunks = document.getElementById('exp-total-chunks');
  const expDimension = document.getElementById('exp-dimension');
  const expSources = document.getElementById('exp-sources');
  const expSize = document.getElementById('exp-size');
  const chunksTableBody = document.getElementById('chunks-table-body');
  const sourcesSummaryContainer = document.getElementById('sources-summary-container');
  const refreshDocsBtn = document.getElementById('refresh-docs-btn');
  const refreshSourcesTableBtn = document.getElementById('refresh-sources-table-btn');
  const clearDocsBtn = document.getElementById('clear-docs-btn');

  function renderSourcesSummary() {
    if (!sourcesSummaryContainer) return;
    if (!activeSourcesList || activeSourcesList.length === 0) {
      sourcesSummaryContainer.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 2rem; text-align: center; color: var(--apple-text-secondary);">
          No papers or websites added yet. Click "Add Papers &amp; Links" above to add your first document.
        </div>
      `;
      return;
    }

    sourcesSummaryContainer.innerHTML = '';
    activeSourcesList.forEach(src => {
      const card = document.createElement('div');
      card.className = 'source-item-card';
      const icon = src.type === 'web' ? '🌐' : (src.type === 'document' ? '📄' : '📝');
      const typeLabel = src.type === 'web' ? 'Website' : (src.type === 'document' ? 'Document' : 'Notes');

      card.innerHTML = `
        <div class="source-item-top">
          <span class="source-icon">${icon}</span>
          <div style="overflow: hidden;">
            <div class="source-item-title" title="${escapeHtml(src.id)}">${escapeHtml(src.title)}</div>
            <div class="source-item-meta">${src.chunks_count} paragraphs saved</div>
          </div>
        </div>
        <div class="source-item-actions">
          <span class="source-type-pill ${src.type}">${typeLabel}</span>
          <button type="button" class="chat-now-btn" data-source="${escapeHtml(src.id)}">💬 Ask Questions</button>
        </div>
      `;

      card.querySelector('.chat-now-btn').addEventListener('click', () => {
        startChatWithSource(src.id);
      });

      sourcesSummaryContainer.appendChild(card);
    });
  }

  async function loadKnowledgeBase() {
    try {
      const statsRes = await fetch(`${API_BASE}/api/stats`);
      if (statsRes.ok) {
        const stats = await statsRes.json();
        if (expTotalChunks) expTotalChunks.textContent = stats.total_chunks || 0;
        if (expDimension) expDimension.textContent = 'High';
        if (expSources) expSources.textContent = stats.total_sources || 0;
        if (expSize) expSize.textContent = `${(stats.storage_size_kb || 0).toFixed(0)} KB`;
      }
    } catch (e) {
      console.warn('Failed to load stats', e);
    }

    await loadSources();

    try {
      const docsRes = await fetch(`${API_BASE}/api/documents?limit=100`);
      if (docsRes.ok) {
        const data = await docsRes.json();
        renderChunksTable(data.documents || []);
      }
    } catch (e) {
      if (chunksTableBody) {
        chunksTableBody.innerHTML = `<tr><td colspan="4" class="text-center-simple">Could not load saved paragraphs.</td></tr>`;
      }
    }
  }

  function renderChunksTable(docs) {
    if (!chunksTableBody) return;
    if (!docs || docs.length === 0) {
      chunksTableBody.innerHTML = `<tr><td colspan="4" class="text-center-simple">No saved files found.</td></tr>`;
      return;
    }

    chunksTableBody.innerHTML = '';
    docs.forEach((doc, i) => {
      const tr = document.createElement('tr');
      const text = doc.text || doc.chunk || '';
      const preview = text.length > 130 ? text.slice(0, 127) + '...' : text;
      const src = doc.metadata?.source || doc.metadata?.original_filename || doc.metadata?.doc_id || 'Document';
      const id = `Piece #${i + 1}`;

      tr.innerHTML = `
        <td><span class="chunk-id-code">${id}</span></td>
        <td>${escapeHtml(preview)}</td>
        <td><strong style="font-size: 0.825rem;">${escapeHtml(src)}</strong></td>
        <td style="font-size: 0.8rem; color: var(--apple-text-secondary);">${text.length} letters</td>
      `;

      tr.style.cursor = 'pointer';
      tr.title = 'Click to read full paragraph';
      tr.addEventListener('click', () => openChunkModal({ chunk: text, metadata: doc.metadata, id: doc.id }));

      chunksTableBody.appendChild(tr);
    });
  }

  if (refreshDocsBtn) refreshDocsBtn.addEventListener('click', loadKnowledgeBase);
  if (refreshSourcesTableBtn) refreshSourcesTableBtn.addEventListener('click', loadKnowledgeBase);

  if (clearDocsBtn) {
    clearDocsBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to remove all saved papers and websites?')) {
        return;
      }

      try {
        const res = await fetch(`${API_BASE}/api/documents`, { method: 'DELETE' });
        if (res.ok) {
          showToast('All saved files removed', 'info');
          await loadKnowledgeBase();
          await checkStatus();
        }
      } catch (err) {
        showToast(`Could not clear files: ${err.message}`, 'error');
      }
    });
  }

  // Initial Execution
  checkStatus();
  loadSources();
});
