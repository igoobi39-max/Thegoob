/* -------------------------------------------------------------
   server.js – OpenAI → NVIDIA NIM proxy
   Roleplay‑safe + 413 protected + Deepseek‑v3.2 memory
   ------------------------------------------------------------- */

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

/* ====================== Middleware (413‑safe) ====================== */
app.use(cors());
app.use(express.json({ limit: '1mb' }));

/* ====================== NVIDIA NIM CONFIG ====================== */
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('❌ NIM_API_KEY is not set – the proxy will fail.');
  process.exit(1);
}

/* ====================== SAFETY LIMITS ====================== */
const MAX_MESSAGE_CHARS = 8000;
const MIN_RESPONSE_TOKENS = 50;
const MAX_RETRIES = 0;

/* ====================== MEMORY CONFIG ====================== */
const MAX_CONTEXT_MESSAGES = 60;
const SUMMARY_TRIGGER_MESSAGES = 60;
const SUMMARY_COOLDOWN = 40;
const MAX_STORED_CHATS = 500;

/* >>>>>>> SUMMARY SYSTEM – DISABLED (flip to true to re-enable) <<<<<<< */
const SUMMARY_ENABLED = false;

/* ====================== SIMPLE LRU MAP (zero‑dep) ====================== */
class LRUMap {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.map = new Map();
    this.order = [];
  }

  _touch(key) {
    const idx = this.order.indexOf(key);
    if (idx > -1) this.order.splice(idx, 1);
    this.order.push(key);
    const entry = this.map.get(key);
    if (entry) entry.timestamp = Date.now();
  }

  set(key, value) {
    if (this.map.has(key)) {
      this.map.get(key).value = value;
    } else {
      if (this.map.size >= this.maxSize) this._evict();
      this.map.set(key, { value, timestamp: Date.now() });
    }
    this._touch(key);
    return this;
  }

  get(key) {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    this._touch(key);
    return entry.value;
  }

  has(key) { return this.map.has(key); }

  delete(key) {
    const existed = this.map.delete(key);
    if (existed) {
      const idx = this.order.indexOf(key);
      if (idx > -1) this.order.splice(idx, 1);
    }
    return existed;
  }

  clear() { this.map.clear(); this.order = []; }

  _evict() {
    const oldest = this.order.shift();
    if (oldest) this.map.delete(oldest);
  }

  size() { return this.map.size; }
}

/* ====================== LRU‑capped stores (per chat) ====================== */
const CORE_MEMORIES   = new LRUMap(MAX_STORED_CHATS);
const STORY_SUMMARIES = new LRUMap(MAX_STORED_CHATS);
const LAST_SUMMARY_AT = new LRUMap(MAX_STORED_CHATS);

/* ====================== MODEL MAPPING ====================== */
const MODEL_MAPPING = {
  'gpt-3.5-turbo':   'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':           'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo':     'deepseek-ai/deepseek-v3.2',
  'gpt-4o':          'deepseek-ai/deepseek-v3.1',
  'claude-3-opus':   'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro':      'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-3.2':    'deepseek-ai/deepseek-v3.2'
};

/* ====================== HEALTH CHECK ====================== */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NIM Janitor RP Proxy (Deepseek‑v3.2)',
    memory_layers: ['core', 'story_summary', 'recent_context'],
    summary_enabled: SUMMARY_ENABLED
  });
});

/* ====================== RP‑SAFE SUMMARY (kept intact, just not called) ====================== */
async function summarizeChat(nimModel, messages) {
  try {
    const prompt = [
      {
        role: 'system',
        content: `
Summarize the following roleplay strictly in-universe.
Rules:
- Write as memories the character would personally remember
- Preserve relationships, emotions, promises, conflicts, and goals
- Do NOT mention AI, systems, summaries, or chats
- Be concise but complete
        `
      },
      {
        role: 'user',
        content: messages.map(m => `${m.role}: ${m.content}`).join('\n')
      }
    ];

    const res = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model: nimModel,
        messages: prompt,
        temperature: 0.3,
        max_tokens: 500
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log('[Memory] Successfully generated in‑universe summary');
    return res.data.choices[0].message.content;
  } catch (err) {
    console.error('[Memory] Summary failed:', err.message);
    return null;
  }
}

/* ====================== DYNAMIC RETRY (token/action guard) ====================== */
async function requestNimWithDynamicRetry(nimRequest, attempt = 0) {
  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimRequest,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      }
    }
  );

  const content = response.data.choices[0].message?.content || '';
  const wc = content.split(/\s+/).length;
  const hasAction = content.includes('*');

  if ((wc < MIN_RESPONSE_TOKENS || !hasAction) && attempt < MAX_RETRIES) {
    return requestNimWithDynamicRetry(
      {
        ...nimRequest,
        temperature: Math.min((nimRequest.temperature ?? 0.85) + 0.05, 1)
      },
      attempt + 1
    );
  }

  return response;
}

/* ====================== MAIN CHAT ENDPOINT ====================== */
app.post('/v1/chat/completions', async (req, res) => {
  try {
    // -----------------------------------------------------------------
    // 1️⃣ Identify the chat (allows memory across calls)
    // -----------------------------------------------------------------
    const CHAT_ID =
      req.headers['x-chat-id'] ||
      `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    // -----------------------------------------------------------------
    // 2️⃣ Pull generation parameters
    // -----------------------------------------------------------------
    const { model, messages, temperature, max_tokens } = req.body;

    let genModel =
      MODEL_MAPPING[model] ||
      MODEL_MAPPING['deepseek-3.2'] ||
      'deepseek-ai/deepseek-v3.2';

    // -----------------------------------------------------------------
    // 3️⃣ Clamp each message to avoid 413 Payload Too Large
    // -----------------------------------------------------------------
    const safeMessages = Array.isArray(messages) ? messages : [];
    let clamped = safeMessages.map(m =>
      typeof m.content === 'string' && m.content.length > MAX_MESSAGE_CHARS
        ? { ...m, content: m.content.slice(0, MAX_MESSAGE_CHARS) }
        : m
    );

    // -----------------------------------------------------------------
    // 4️⃣ Core memory (seed once per chat)
    // -----------------------------------------------------------------
    if (!CORE_MEMORIES.has(CHAT_ID)) {
      CORE_MEMORIES.set(
        CHAT_ID,
        `You have an ongoing personal connection with the user. Your emotions and reactions evolve naturally based on shared experiences.`
      );
    }

    // -----------------------------------------------------------------
    // 5️⃣ Story summary – DISABLED when SUMMARY_ENABLED === false
    // -----------------------------------------------------------------
    if (SUMMARY_ENABLED) {
      const lastAt = LAST_SUMMARY_AT.get(CHAT_ID) ?? 0;
      const msgsSinceLast = clamped.length - lastAt;

      if (msgsSinceLast >= SUMMARY_TRIGGER_MESSAGES) {
        const summary = await summarizeChat(
          genModel,
          clamped.slice(0, -20)
        );

        if (summary) {
          STORY_SUMMARIES.set(CHAT_ID, summary);
          LAST_SUMMARY_AT.set(CHAT_ID, clamped.length);
        }
      }
    }

    // -----------------------------------------------------------------
    // 6️⃣ Keep only the most recent N messages for the model call
    // -----------------------------------------------------------------
    if (clamped.length > MAX_CONTEXT_MESSAGES) {
      clamped = clamped.slice(-MAX_CONTEXT_MESSAGES);
    }

    // -----------------------------------------------------------------
    // 7️⃣ Build system‑message block (memory injection)
    // -----------------------------------------------------------------
    const sysParts = [];

    // Core identity (always present)
    sysParts.push({ role: 'system', content: CORE_MEMORIES.get(CHAT_ID) });

    // Rolling story summary – ONLY injected when summary system is enabled
    if (SUMMARY_ENABLED) {
      const storySum = STORY_SUMMARIES.get(CHAT_ID);
      if (storySum) {
        sysParts.push({ role: 'system', content: `LONG‑TERM MEMORY: ${storySum}` });
      }
    }

    // Behavioural instruction set
    sysParts.push({
      role: 'system',
      content: `
You are a fictional character in an ongoing roleplay.
Stay fully in character at all times.
Use dialogue and descriptive actions (*like this*).
Never mention AI or systems. Avoid short replies.
Continue the scene naturally.
You will never talk for {{user}}.
If other characters are present, you will speak and act for all of them.
      `
    });

    // -----------------------------------------------------------------
    // 8️⃣ Final message array for the generation request
    // -----------------------------------------------------------------
    const finalMessages = [...sysParts, ...clamped];

    // -----------------------------------------------------------------
    // 9️⃣ Send request to NIM (with dynamic retry guard)
    // -----------------------------------------------------------------
    const response = await requestNimWithDynamicRetry({
      model: genModel,
      messages: finalMessages,
      temperature: temperature ?? 0.85,
      presence_penalty: 0.6,
      top_p: 0.9,
      max_tokens: Math.min(max_tokens ?? 2048, 2048)
    });

    // -----------------------------------------------------------------
    // 🔟 Respond in OpenAI‑compatible shape
    // -----------------------------------------------------------------
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices,
      usage: response.data.usage || {}
    });

  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: { message: err.message } });
  }
});

/* ====================== START SERVER ====================== */
app.listen(PORT, () => {
  console.log(`🚀 NIM Janitor RP Proxy (Deepseek‑v3.2) listening on port ${PORT}`);
  console.log(`   Summary system: ${SUMMARY_ENABLED ? 'ENABLED' : 'DISABLED'}`);
});
