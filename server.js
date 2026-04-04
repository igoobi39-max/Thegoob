// server.js - OpenAI to NVIDIA NIM API Proxy
// Janitor RP Safe + 413 Protected + OpenRouter-like Layer
// + Dynamic Auto-Regeneration + Multi-Layer Per-Chat Memory

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ======================
//  Middleware (413 SAFE)
// ======================
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ======================
//  NVIDIA NIM CONFIG
// ======================
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ======================
//  SAFE LIMITS & CONFIG
// ======================
const MAX_MESSAGE_CHARS = 8000;
const MIN_RESPONSE_TOKENS = 50;
const MAX_RETRIES = 2;

// ======================
//  SMART MEMORY CONFIG
// ======================
const MAX_CONTEXT_MESSAGES = 30;     // Dropped from 40 to 30 to keep chats fast
const SUMMARY_TRIGGER_MESSAGES = 80; // Wait longer (80 messages) to start
const SUMMARY_COOLDOWN = 60;         // Wait 60 messages between updates

// ======================
//  MEMORIES STORAGE (PER CHAT)
// ======================
const CORE_MEMORIES = new Map();        // Stable identity memory
const STORY_SUMMARIES = new Map();      // Rolling plot summary
const LAST_SUMMARY_AT = new Map();      // Cooldown tracker

// ======================
//  MODEL MAPPING
// ======================
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// ======================
//  HEALTH CHECK
// ======================
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NIM Janitor RP Proxy',
    memory_layers: ['core', 'story_summary', 'recent_context']
  });
});

// ======================
//  HELPER: RP-SAFE SUMMARY (HIGH COMPRESSION)
// ======================
async function summarizeChat(nimModel, messages, existingSummary = "") {
  try {
    const prompt = [
      {
        role: 'system',
        content: `
Analyze the roleplay and produce a ultra-dense, compact summary.
Incorporate any existing summary context.
Use strict in-universe shorthand. No meta-talk or complete sentences.

Format exactly like this:
LOC: [Current setting]
RELATION: [1-sentence status of bond]
KEY PLOT: [Core conflict/event]
GOALS: [What they are trying to do next]
`
      },
      {
        role: 'user',
        content: `EXISTING CONTEXT: ${existingSummary}\n\nNEW EVENTS TO ADD:\n` + 
                 messages.map(m => `${m.role}: ${m.content}`).join('\n')
      }
    ];

    const res = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model: nimModel,
        messages: prompt,
        temperature: 0.2, // Low temp for focused, robotic extraction
        max_tokens: 150   // Hard cap at 150 tokens to save resources
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log(`[Memory] Compact summary generated.`);
    return res.data.choices[0].message.content;
  } catch (err) {
    console.error('[Memory] Summary failed:', err.message);
    return null;
  }
}

// ======================
//  HELPER: AUTO-RETRY
// ======================
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
      { ...nimRequest, temperature: Math.min((nimRequest.temperature ?? 0.85) + 0.05, 1) },
      attempt + 1
    );
  }

  return response;
}

// ======================
//  CHAT COMPLETIONS
// ======================
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const CHAT_ID =
      req.headers['x-chat-id'] ||
      `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    const { model, messages, temperature, max_tokens } = req.body;

    let nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.1-terminus';

    // Clamp message lengths
    let safeMessages = Array.isArray(messages) ? messages : [];
    safeMessages = safeMessages.map(m =>
      typeof m?.content === 'string' && m.content.length > MAX_MESSAGE_CHARS
        ? { ...m, content: m.content.slice(0, MAX_MESSAGE_CHARS) }
        : m
    );

    // ======================
    //  CORE MEMORY (SEED ONCE)
    // ======================
    if (!CORE_MEMORIES.has(CHAT_ID)) {
      CORE_MEMORIES.set(
        CHAT_ID,
        `
You have an ongoing personal connection with the user.
Your emotions and reactions evolve naturally based on shared experiences.
`
      );
    }

    // ======================
    //  STORY SUMMARY (ROLLING - COMPACTED)
    // ======================
    const lastAt = LAST_SUMMARY_AT.get(CHAT_ID) || 0;

    if (
      safeMessages.length > SUMMARY_TRIGGER_MESSAGES &&
      safeMessages.length - lastAt >= SUMMARY_COOLDOWN
    ) {
      // Only send the middle chunk of messages to be summarized
      const messagesToSummarize = safeMessages.slice(lastAt, -20);
      const existingSummary = STORY_SUMMARIES.get(CHAT_ID) || "";
      
      const summary = await summarizeChat(
        nimModel,
        messagesToSummarize,
        existingSummary
      );

      if (summary) {
        STORY_SUMMARIES.set(CHAT_ID, summary);
        LAST_SUMMARY_AT.set(CHAT_ID, safeMessages.length);
      }
    }

    // ======================
    //  CONTEXT TRIMMING
    // ======================
    if (safeMessages.length > MAX_CONTEXT_MESSAGES) {
      safeMessages = safeMessages.slice(-MAX_CONTEXT_MESSAGES);
    }

    // ======================
    //  MEMORY INJECTION
    // ======================
    const memoryInjection = [
      { role: 'system', content: CORE_MEMORIES.get(CHAT_ID) },
      STORY_SUMMARIES.has(CHAT_ID) 
        ? { role: 'system', content: `LONG-TERM MEMORY:\n${STORY_SUMMARIES.get(CHAT_ID)}` }
        : null,
      {
        role: 'system',
        content: `
You are a fictional character in an ongoing roleplay.
Stay fully in character at all times.
Use dialogue and descriptive actions (*like this*).
Never mention AI or systems.
Avoid short replies. Continue the scene naturally.
You will never talk for {{user}}
If there other characters present in a scene, you will talk and act for all of them
`
      }
    ].filter(Boolean);

    safeMessages = [...memoryInjection, ...safeMessages];

    // ======================
    //  SEND REQUEST
    // ======================
    const response = await requestNimWithDynamicRetry({
      model: nimModel,
      messages: safeMessages,
      temperature: temperature ?? 0.85,
      presence_penalty: 0.6,
      top_p: 0.9,
      max_tokens: Math.min(max_tokens ?? 2048, 2048)
    });

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

// ======================
//  START SERVER
// ======================
app.listen(PORT, () => {
  console.log(`NIM Janitor RP Proxy running on port ${PORT}`);
});
