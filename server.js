/* -------------------------------------------------------------
   server.js – OpenAI → NVIDIA NIM proxy
   Roleplay-safe + 413 protected + Deepseek-v3.2
   NOW WITH PROPER SSE STREAMING
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
const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('❌ NIM_API_KEY is not set – the proxy will fail.');
  process.exit(1);
}

/* ====================== SAFETY LIMITS ====================== */
const MAX_MESSAGE_CHARS = 8000;
const MIN_RESPONSE_TOKENS = 50;
const MAX_RETRIES = 0;
const NIM_TIMEOUT = 120_000; // 2 min hard ceiling per request

/* ====================== MEMORY CONFIG ====================== */
const MAX_CONTEXT_MESSAGES = 60;
const SUMMARY_TRIGGER_MESSAGES = 60;
const SUMMARY_COOLDOWN = 40;
const MAX_STORED_CHATS = 500;
const SUMMARY_ENABLED = false;

/* ====================== SIMPLE LRU MAP (zero‑dep) ====================== */
class LRUMap {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.map = new Map();
    this.order = []; // most-recent at end
  }
  set(key, value) {
    if (this.map.has(key)) {
      this.map.get(key).value = value;
      this._touch(key);
      return this;
    }
    if (this.map.size >= this.maxSize) this._evict();
    this.map.set(key, { value, timestamp: Date.now() });
    this.order.push(key);
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
  size() { return this.map.size; }
  _touch(key) {
    const idx = this.order.indexOf(key);
    if (idx > -1) this.order.splice(idx, 1);
    this.order.push(key);
  }
  _evict() {
    const oldest = this.order.shift();
    if (oldest) this.map.delete(oldest);
  }
}

/* ====================== LRU‑capped stores ====================== */
const CORE_MEMORIES  = new LRUMap(MAX_STORED_CHATS);
const STORY_SUMMARIES = new LRUMap(MAX_STORED_CHATS);
const LAST_SUMMARY_AT = new LRUMap(MAX_STORED_CHATS);

/* ====================== MODEL MAPPING ====================== */
const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':          'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo':    'deepseek-ai/deepseek-v3.2',
  'gpt-4o':         'deepseek-ai/deepseek-v3.1',
  'claude-3-opus':  'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro':     'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-3.2':   'deepseek-ai/deepseek-v3.2'
};

/* ====================== HEALTH CHECK ====================== */
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'NIM Janitor RP Proxy (Deepseek-v3.2) + Streaming',
    summary_enabled: SUMMARY_ENABLED
  });
});

/* ====================== RP‑SAFE SUMMARY (disabled) ====================== */
async function summarizeChat(nimModel, messages) {
  try {
    const prompt = [
      { role: 'system', content:
        `Summarize the following roleplay strictly in-universe.
         Rules: Write as memories the character would personally remember.
         Preserve relationships, emotions, promises, conflicts, and goals.
         Do NOT mention AI, systems, summaries, or chats. Be concise but complete.`
      },
      { role: 'user', content:
        messages.map(m => `${m.role}: ${m.content}`).join('\n')
      }
    ];
    const res = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model: nimModel, messages: prompt, temperature: 0.3, max_tokens: 500 },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: NIM_TIMEOUT
      }
    );
    console.log('[Memory] Summary generated');
    return res.data.choices[0].message.content;
  } catch (err) {
    console.error('[Memory] Summary failed:', err.message);
    return null;
  }
}

/* ============================================================
   HELPER – build the final messages array (shared by both
   streaming and non-streaming paths)
   ============================================================ */
async function buildFinalMessages(reqBody) {
  const { model, messages, temperature, max_tokens } = reqBody;

  const genModel =
    MODEL_MAPPING[model] ||
    MODEL_MAPPING['deepseek-3.2'] ||
    'deepseek-ai/deepseek-v3.2';

  /* ---- clamp individual messages ---- */
  let clamped = (Array.isArray(messages) ? messages : []).map(m =>
    typeof m.content === 'string' && m.content.length > MAX_MESSAGE_CHARS
      ? { ...m, content: m.content.slice(0, MAX_MESSAGE_CHARS) }
      : m
  );

  /* ---- core memory (seed once per chat) ---- */
  const CHAT_ID =
    reqBody.chat_id ||
    `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  if (!CORE_MEMORIES.has(CHAT_ID)) {
    CORE_MEMORIES.set(
      CHAT_ID,
      `You have an ongoing personal connection with the user. Your emotions and reactions evolve naturally based on shared experiences.`
    );
  }

  /* ---- story summary (disabled) ---- */
  if (SUMMARY_ENABLED) {
    const lastAt = LAST_SUMMARY_AT.get(CHAT_ID) ?? 0;
    const msgsSince = clamped.length - lastAt;
    if (msgsSince >= SUMMARY_TRIGGER_MESSAGES) {
      const summary = await summarizeChat(genModel, clamped.slice(0, -20));
      if (summary) {
        STORY_SUMMARIES.set(CHAT_ID, summary);
        LAST_SUMMARY_AT.set(CHAT_ID, clamped.length);
      }
    }
  }

  /* ---- trim to recent context ---- */
  if (clamped.length > MAX_CONTEXT_MESSAGES) {
    clamped = clamped.slice(-MAX_CONTEXT_MESSAGES);
  }

  /* ---- build system block ---- */
  const sysParts = [];
  sysParts.push({ role: 'system', content: CORE_MEMORIES.get(CHAT_ID) });

  if (SUMMARY_ENABLED) {
    const storySum = STORY_SUMMARIES.get(CHAT_ID);
    if (storySum) {
      sysParts.push({ role: 'system', content: `LONG-TERM MEMORY: ${storySum}` });
    }
  }

  sysParts.push({
    role: 'system',
    content: `You are a fictional character in an ongoing roleplay. Stay fully in character at all times.
Use dialogue and descriptive actions (*like this*). Never mention AI or systems.
Avoid short replies. Continue the scene naturally.
You will never talk for {{user}}. If other characters are present, you will speak and act for all of them.`
  });

  const finalMessages = [...sysParts, ...clamped];

  /* ---- build NIM request body (pass through Janitor's params) ---- */
  const nimBody = {
    model: genModel,
    messages: finalMessages,
    temperature:  temperature ?? 0.85,
    top_p:        reqBody.top_p   ?? 0.9,
    frequency_penalty: reqBody.frequency_penalty ?? 0,
    presence_penalty:  reqBody.presence_penalty  ?? 0.6,
    max_tokens:   Math.min(max_tokens ?? 2048, 2048)
  };

  return { nimBody, genModel, CHAT_ID, requestedModel: model };
}

/* ============================================================
   NON-STREAMING PATH (kept as fallback)
   ============================================================ */
async function handleNonStreaming(reqBody, res) {
  const { nimBody, requestedModel } = await buildFinalMessages(reqBody);

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimBody,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: NIM_TIMEOUT
    }
  );

  /* ---- dynamic retry (same as before) ---- */
  let content = response.data.choices[0].message?.content || '';
  let wc = content.split(/\s+/).length;
  let attempt = 0;

  while ((wc < MIN_RESPONSE_TOKENS || !content.includes('*')) && attempt < MAX_RETRIES) {
    nimBody.temperature = Math.min((nimBody.temperature ?? 0.85) + 0.05, 1);
    const retryRes = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimBody,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: NIM_TIMEOUT
      }
    );
    content = retryRes.data.choices[0].message?.content || '';
    wc = content.split(/\s+/).length;
    attempt++;
  }

  res.json({
    id:      `chatcmpl-${Date.now()}`,
    object:  'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model:   requestedModel,
    choices: response.data.choices,
    usage:   response.data.usage || {}
  });
}

/* ============================================================
   ★ STREAMING PATH ★  — this is what makes it fast
   ============================================================ */
async function handleStreaming(reqBody, res) {
  const { nimBody, requestedModel } = await buildFinalMessages(reqBody);

  // Tell NIM we want a stream
  nimBody.stream = true;
  nimBody.stream_options = { include_usage: true };

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimBody,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept':       'text/event-stream'   // hint for NIM
      },
      timeout: NIM_TIMEOUT,
      responseType: 'stream'                   // ★ key line
    }
  );

  // Set SSE headers for Janitor AI
  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control',  'no-cache');
  res.setHeader('Connection',     'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');   // nginx fix

  // Pipe NIM's stream straight to the client
  response.data.on('data', (chunk) => {
    res.write(chunk);
  });

  response.data.on('end', () => {
    res.end();
  });

  response.data.on('error', (err) => {
    console.error('[Stream] NIM stream error:', err.message);
    // Try to send a proper SSE error event
    try {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    } catch (_) {}
    res.end();
  });

  // If client disconnects, kill the upstream
  req.on('close', () => {
    response.data.destroy();
  });
}

/* ============================================================
   MAIN ENDPOINT – dispatch to streaming or non-streaming
   ============================================================ */
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (req.body.stream === true) {
      await handleStreaming(req.body, res);
    } else {
      await handleNonStreaming(req.body, res);
    }
  } catch (err) {
    console.error('Proxy error:', err.message);
    if (!res.headersSent) {
      res.status(500).json({ error: { message: err.message } });
    } else {
      // Headers already sent (streaming started) – just end the stream
      try { res.write(`data: [DONE]\n\n`); } catch (_) {}
      res.end();
    }
  }
});

/* ====================== START SERVER ====================== */
app.listen(PORT, () => {
  console.log(`🚀 NIM Janitor RP Proxy + Streaming on port ${PORT}`);
  console.log(`   Summary system: ${SUMMARY_ENABLED ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   Streaming:      ENABLED`);
});
