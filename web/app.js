/**
 * SpaceAX AI v3.0 — Client App JS
 * Handles Chat UI, Mention/Highlight Follow-up, Vision Upload, Provider Switching & Agent Mode
 */

document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const providerSelect = document.getElementById('provider-select');
  const endpointConfig = document.getElementById('endpoint-config');
  const apiEndpointUrl = document.getElementById('api-endpoint-url');
  const modelStatusName = document.getElementById('model-status-name');
  const imageUpload = document.getElementById('image-upload');
  const imagePreviewContainer = document.getElementById('image-preview-container');
  const imagePreview = document.getElementById('image-preview');
  const removeImageBtn = document.getElementById('remove-image-btn');
  const mentionBar = document.getElementById('mention-bar');
  const mentionText = document.getElementById('mention-text');
  const cancelMentionBtn = document.getElementById('cancel-mention');
  const toggleAgentBtn = document.getElementById('toggle-agent-mode');
  const toggleWebBtn = document.getElementById('toggle-web-mode');
  const clearChatBtn = document.getElementById('clear-chat');
  const newChatBtn = document.getElementById('new-chat-btn');

  let currentMention = null;
  let currentImageBase64 = null;
  let isAgentMode = false;
  let isWebSearchMode = true;

  // Toggle Web Search Mode
  if (toggleWebBtn) {
    toggleWebBtn.addEventListener('click', () => {
      isWebSearchMode = !isWebSearchMode;
      toggleWebBtn.classList.toggle('active', isWebSearchMode);
      toggleWebBtn.textContent = `Web Search: ${isWebSearchMode ? 'ON' : 'OFF'}`;
    });
  }

  // Auto-resize textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  });

  // Handle Provider Selection
  providerSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    modelStatusName.textContent = `Engine: ${e.target.options[e.target.selectedIndex].text}`;
    if (val === 'ollama' || val === 'lmstudio' || val === 'openai') {
      endpointConfig.classList.remove('hidden');
      if (val === 'ollama') apiEndpointUrl.value = 'http://localhost:11434/api/generate';
      if (val === 'lmstudio') apiEndpointUrl.value = 'http://localhost:1234/v1/chat/completions';
      if (val === 'openai') apiEndpointUrl.value = 'https://api.openai.com/v1/chat/completions';
    } else {
      endpointConfig.classList.add('hidden');
    }
  });

  // Toggle Agent Mode
  toggleAgentBtn.addEventListener('click', () => {
    isAgentMode = !isAgentMode;
    toggleAgentBtn.classList.toggle('active', isAgentMode);
    toggleAgentBtn.textContent = `Agent Loop: ${isAgentMode ? 'ON' : 'OFF'}`;
  });

  // Text Highlight to Mention / Reply
  document.addEventListener('mouseup', () => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
    if (selectedText.length > 5) {
      setMention(selectedText);
    }
  });

  function setMention(text) {
    currentMention = text;
    mentionText.textContent = `"${text}"`;
    mentionBar.classList.remove('hidden');
  }

  cancelMentionBtn.addEventListener('click', () => {
    currentMention = null;
    mentionBar.classList.add('hidden');
  });

  // Image Upload Handling
  imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        currentImageBase64 = event.target.result;
        imagePreview.src = currentImageBase64;
        imagePreviewContainer.classList.remove('hidden');
      };
      reader.readAsDataURL(file);
    }
  });

  removeImageBtn.addEventListener('click', () => {
    currentImageBase64 = null;
    imageUpload.value = '';
    imagePreviewContainer.classList.add('hidden');
  });

  // Clear Chat
  clearChatBtn.addEventListener('click', clearChat);
  newChatBtn.addEventListener('click', clearChat);

  function clearChat() {
    chatMessages.innerHTML = `
      <div class="welcome-card">
        <div class="welcome-icon">✨</div>
        <h2>SpaceAX AI v3.0</h2>
        <p>Kecerdasan buatan mandiri dengan Vision, MoE, Agent System, dan Bahasa Indonesia Alami.</p>
      </div>`;
  }

  // Handle Form Submit
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text && !currentImageBase64) return;

    // Hide welcome card if present
    const welcome = chatMessages.querySelector('.welcome-card');
    if (welcome) welcome.remove();

    // Prepare User Message
    let fullUserMsg = text;
    if (currentMention) {
      fullUserMsg = `[Replying to: "${currentMention}"]\n${text}`;
    }

    appendMessage('user', fullUserMsg, currentImageBase64);

    // Reset Input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    const activeMention = currentMention;
    const activeImage = currentImageBase64;

    currentMention = null;
    mentionBar.classList.add('hidden');
    removeImageBtn.click();

    // Show AI Thinking Indicator
    const thinkingId = appendThinking();

    try {
      const provider = providerSelect.value;
      let aiResponseText = '';

      if (isAgentMode) {
        // Agent Mode Execution
        const res = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: text })
        });
        const data = await res.json();
        aiResponseText = data.response || 'Agent execution completed.';
      } else if (provider.startsWith('spaceax')) {
        // Internal SpaceAX Model
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: provider,
            web_enabled: isWebSearchMode,
            messages: [{ role: 'user', content: fullUserMsg }]
          })
        });
        const data = await res.json();
        aiResponseText = data.choices[0].message.content;
      } else {
        // External Provider (Ollama / LM Studio / OpenAI)
        const endpoint = apiEndpointUrl.value;
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'default',
            messages: [{ role: 'user', content: fullUserMsg }],
            prompt: fullUserMsg,
            stream: false
          })
        });
        const data = await res.json();
        aiResponseText = data.response || (data.choices && data.choices[0].message.content) || JSON.stringify(data);
      }

      removeThinking(thinkingId);
      appendMessage('ai', aiResponseText);
    } catch (err) {
      removeThinking(thinkingId);
      appendMessage('ai', `⚠️ Error: Gagal mendapatkan respon dari AI engine (${err.message}). Pastikan server aktif.`);
    }
  });

  // UI Helpers
  function appendMessage(role, text, imgSrc = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AX';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (imgSrc) {
      const img = document.createElement('img');
      img.src = imgSrc;
      img.style.maxWidth = '200px';
      img.style.borderRadius = '8px';
      img.style.display = 'block';
      img.style.marginBottom = '8px';
      bubble.appendChild(img);
    }

    const textSpan = document.createElement('span');
    textSpan.innerHTML = text.replace(/\n/g, '<br/>');
    bubble.appendChild(textSpan);

    row.appendChild(avatar);
    row.appendChild(bubble);

    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendThinking() {
    const id = 'thinking-' + Date.now();
    const row = document.createElement('div');
    row.id = id;
    row.className = 'message-row ai';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = 'AX';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = 'Thinking...';

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }
});

// Quick prompt helper
function sendQuickPrompt(promptText) {
  const input = document.getElementById('chat-input');
  input.value = promptText;
  document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}
