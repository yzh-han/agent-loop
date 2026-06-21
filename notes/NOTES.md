# Chat Completions API 核心笔记

> 环境：vLLM `http://localhost:8000` 托管 `gemma-4-E4B-it`
> 运行：`uv run python src/agent/demo_chat.py` | `uv run python src/agent/demo_messages.py` | `uv run python -m agent.run_agent -q "北京天气？"`

---

## 0. OpenAI 是什么？

### 0.1 公司 & API 标准

**OpenAI** 是 2015 年成立的 AI 公司，发布了 GPT 系列模型和 ChatGPT。

但对你来说更重要的不是公司本身，而是它**定义了一套 API 协议**，现在已经成为整个 LLM 行业的**事实标准**。无论你用的是：

| 推理框架 | 模型 | 兼容 OpenAI API？ |
|----------|------|------------------|
| vLLM | Gemma / LLaMA / Qwen / ... | ✅ `/v1/chat/completions` |
| Ollama | LLaMA / Mistral / ... | ✅ |
| llama.cpp | LLaMA / Gemma / ... | ✅ 需启动 server |
| 真正的 OpenAI | GPT-4o / GPT-5 / ... | ✅ 原生 |

**所有框架都实现了同一套 API 协议。** 学会这一个 SDK，就能调用任何 LLM。

### 0.2 `openai` Python SDK

```bash
pip install openai
```

这个库不是"真正的 OpenAI 专属"——它本质上是一个 **HTTP client + 数据模型封装**。

```
你写的代码                   openai SDK                       HTTP
─────────                   ──────────                       ────
client.chat.completions    序列化请求体
    .create(               添加默认参数
        model=...,         构造 auth header        ──►  POST /v1/chat/completions
        messages=[...],    处理 stream/retry              Host: localhost:8000
        tools=[...],       ...                           Body: { "model": "...",
    )                                                           "messages": [...] }
        │                                                       │
        ▼                                                       ▼
ChatCompletion 对象   ◄──  解析 JSON 响应 ◄──  {"id":"chatcmpl-xxx",
 (pydantic model)          处理 SSE 流              "choices":[...],
                           抛异常（状态码≠200）       "usage":{...}}
```

### 0.3 核心方法速查

```python
from openai import OpenAI

# 创建 client —— base_url 指向谁就用谁的 API
client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM / 任何兼容服务
    # base_url="https://api.openai.com/v1",  # 真实的 OpenAI
    api_key="not-needed",  # vLLM 不需要；真实 OpenAI 用 sk-xxx
)
```

#### ① `client.chat.completions.create()` —— 对话补全（最常用）

```python
response = client.chat.completions.create(
    model="gemma-4-E4B-it",          # 模型名
    messages=[                        # 对话历史
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好！"},
    ],
    temperature=0.7,                  # 采样温度 (0~2)，越高越随机
    max_tokens=1024,                  # 最多生成多少 token
    top_p=0.95,                       # nucleus sampling 阈值
    stop=["\n"],                      # 遇到这些字符串就停
    stream=False,                     # True=逐 token 流式返回
    tools=[...],                      # 工具定义（见第 5 节）
    tool_choice="auto",               # 工具选择策略: auto / none / required
)

# 响应结构
response.id                           # "chatcmpl-xxx"
response.model                        # "gemma-4-E4B-it"
response.choices[0].message.role      # "assistant"
response.choices[0].message.content   # 模型文本回复
response.choices[0].message.tool_calls  # 工具调用（如果有）
response.choices[0].finish_reason     # "stop" | "tool_calls" | "length"
response.usage.prompt_tokens          # prompt 用了多少 token
response.usage.completion_tokens      # 生成用了多少 token
response.usage.total_tokens           # 总共
```

#### ② `client.chat.completions.create(stream=True)` —— 流式

```python
stream = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,  # ← 关键参数
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)

# chunk.id                            # "chatcmpl-xxx"（每个 chunk 相同）
# chunk.choices[0].delta.role         # "assistant"（只有第一个 chunk 有）
# chunk.choices[0].delta.content      # 本次新增的文本片段
# chunk.choices[0].finish_reason      # None → 最后是 "stop"
```

#### ③ `client.responses.create()` —— Responses API（OpenAI 新 API）

```python
response = client.responses.create(
    model="gemma-4-E4B-it",
    input=[                             # ← 用 input 不用 messages
        {"type": "message", "role": "user", "content": "你好"},
    ],
    max_output_tokens=512,              # ← 参数名也不同
    tools=[...],                        # 内置工具: web_search, code_interpreter 等
)

# response.output                       # 输出 item 列表，每个有 type 字段
# response.output[0].content[0].text    # 文本回复
# response.usage.input_tokens           # 用量字段名也不同
```

#### ④ `client.models.list()` —— 列出可用模型

```python
models = client.models.list()
for m in models.data:
    print(m.id)  # "gemma-4-E4B-it"
```

#### ⑤ 其他 API

```python
# Embeddings —— 文本转向量
client.embeddings.create(model="xxx", input=["你好"])

# Tokenize（vLLM 扩展 API，在 /tokenize）
client.post("/tokenize", body={"model": "xxx", "messages": [...]})

# Completions（旧版 API，不推荐）
client.completions.create(model="xxx", prompt="你好")
```

### 0.4 SDK vs 裸 HTTP

```python
# ── SDK 写法 ──
response = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)

# ── 等价裸 HTTP ──
import httpx, json
resp = httpx.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "你好"}],
})
data = resp.json()
print(data["choices"][0]["message"]["content"])
```

**SDK 帮你做的事：**
1. JSON 序列化/反序列化（dict ↔ Pydantic model）
2. 流式 SSE 解析（`data: {...}\n\n` → Python 对象）
3. 错误处理（4xx/5xx → Python 异常）
4. 重试和超时
5. Type hint / IDE 自动补全（因为返回是 Pydantic model 而不是 dict）

### 0.5 `openai.types.chat` —— 类型系统

SDK 把 API 返回的 JSON **映射成**带类型标注的 Python 对象。所有类型定义在 `openai/types/chat/` 目录下，每个文件是一个 Pydantic model。

**核心认知：** 你写的 `response.choices[0].message.content` 能有 IDE 自动补全，就是因为 `response` 是 `ChatCompletion` 对象，`.message` 是 `ChatCompletionMessage` 对象——每个字段都有明确的类型。

#### 两类类型：Param vs 无后缀

| 后缀 | 方向 | 你在代码里的角色 |
|------|------|----------------|
| **无后缀** | 响应（你**收到**的） | `response.choices[0].message` |
| **`*Param`** | 请求（你**发出**的） | `messages=[...]` 里每个元素 |

```python
from openai.types.chat import ChatCompletion          # 响应类型——response 本身
from openai.types.chat import ChatCompletionMessage   # 响应类型——response.choices[0].message
from openai.types.chat import ChatCompletionChunk     # 响应类型——流式 chunk
from openai.types.chat import ChatCompletionMessageParam  # 请求类型——messages 元素
from openai.types.chat import ChatCompletionToolParam     # 请求类型——tools 元素
```

#### 响应类型树（你收到的东西）

```
ChatCompletion                          ← response 本身
├── id: str                             ← "chatcmpl-xxx"
├── object: "chat.completion"
├── model: str
├── choices: List[Choice]
│   ├── index: int
│   ├── finish_reason: "stop" | "length" | "tool_calls" | "content_filter"
│   └── message: ChatCompletionMessage
│       ├── role: "assistant"           ← 永远是 assistant
│       ├── content: str | None         ← 文本；tool_call 时为 None
│       ├── refusal: str | None         ← 安全拒绝
│       ├── tool_calls: List[ChatCompletionMessageFunctionToolCall] | None
│       │   ├── id: str                 ← "call_abc123"
│       │   ├── type: "function"
│       │   └── function: Function
│       │       ├── name: str           ← "get_weather"
│       │       └── arguments: str      ← '{"city":"北京"}'（注意是 JSON string）
│       ├── annotations: List[Annotation] | None    ← web search 引用标注
│       └── audio: ChatCompletionAudio | None       ← 语音输出
│
├── usage: CompletionUsage
│   ├── prompt_tokens: int
│   ├── completion_tokens: int
│   └── total_tokens: int
│
└── system_fingerprint: str | None

ChatCompletionChunk                     ← 流式 chunk（与上面不同！）
├── id, model, created                 ← 同上
├── choices: List[Choice]
│   └── delta: ChoiceDelta             ← 不是 message，是 delta（增量）
│       ├── role: str | None           ← 仅第一个 chunk 有
│       ├── content: str | None        ← 本次新增的文字
│       └── tool_calls: List[...] | None
└── usage: CompletionUsage | None      ← 仅最后一个 chunk 有
```

#### Content Part 类型（多模态消息的元素）

当 `content` 是 list 时（图片/音频等），每项的类型：

```
ChatCompletionContentPartText           ← {"type": "text", "text": "..."}
├── type: "text"
└── text: str

ChatCompletionContentPartImage          ← {"type": "image_url", "image_url": {...}}
├── type: "image_url"
└── image_url: ImageURL
    ├── url: str                        ← https:// 或 data:image/jpeg;base64,...
    └── detail: "auto" | "low" | "high"

ChatCompletionContentPartInputAudio     ← {"type": "input_audio", ...}
```

#### 为什么 Param 和无后缀要分开？

因为**请求和响应的数据格式不完全对称**：

```python
# 请求时 role 有 5 种可能值
ChatCompletionMessageParam.role:
    Literal["system", "user", "assistant", "tool", "developer"]

# 但响应时 assistant 消息的 role 永远是 "assistant"
ChatCompletionMessage.role:
    Literal["assistant"]  # ← 只有一种！

# 流式和非流式的响应结构也不同：
# 非流式用 ChatCompletion.choices[0].message     ← message 是完整对象
# 流式用  ChatCompletionChunk.choices[0].delta    ← delta 是增量片段
```

#### 实际使用时的类型流转

```python
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

# ── 非流式：返回 ChatCompletion ──
response: ChatCompletion = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "你好"}],
)
msg: ChatCompletionMessage = response.choices[0].message
# msg.content          → IDE 知道是 str | None
# msg.tool_calls       → IDE 知道是 List[...] | None
# msg.tool_calls[0].id → IDE 知道是 str
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(tc.function.name)  # ← 自动补全

# ── 流式：返回 Stream[ChatCompletionChunk] ──
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:                        # chunk: ChatCompletionChunk
    delta = chunk.choices[0].delta           # delta: ChoiceDelta（不是 message！）
    if delta.content:
        print(delta.content, end="")        # 逐字打印
    if chunk.choices[0].finish_reason:
        print(f"\nfinish_reason={chunk.choices[0].finish_reason}")
```

**一句话：`openai.types.chat` 就是把 API 的 JSON 协议翻译成 Python class 体系。底层是 Pydantic `BaseModel`，所以有类型检查、IDE 补全、`.model_dump()` 序列化等能力。**

---

## 1. 总览：一条请求的完整旅程

```
你的代码                      vLLM API Server               GPU 推理引擎
───────                      ───────────────               ────────────
messages[]  ──JSON──►   chat_template 拼接  ──token_ids──►  forward pass
                         tokenizer.encode()                逐 token 采样
                         SamplingParams                    ◄── token_ids ──
                         ◄── SSE streaming ──
for chunk in stream:
    print(chunk)
```

---

## 2. 最简请求 → 完整 JSON

### 2.1 非流式（一次拿完）

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

response = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[
        {"role": "system", "content": "你是精炼的助手。"},
        {"role": "user", "content": "什么是 token？"},
    ],
    temperature=0.7,
    max_tokens=512,
    stream=False,
)
print(response.choices[0].message.content)
```

### 2.2 Request Body —— 全部参数

```json5
{
  // ── 必填 ──
  "model": "gemma-4-E4B-it",            // 模型名（必填）
  "messages": [                          // 对话历史（必填）
    {"role": "system", "content": "你是精炼的助手。"},
    {"role": "user", "content": "什么是 token？"}
  ],

  // ── 采样控制 ──
  "temperature": 0.7,                   // 0~2，越高越随机；默认 1
  "top_p": 0.95,                        // nucleus sampling，和 temperature 二选一
  "seed": 42,                           // 设置后尽量确定性输出（beta）
  "stop": ["\n", "###"],               // 最多 4 个停止词

  // ── 长度控制 ──
  "max_tokens": 512,                    // ⚠️ 已废弃，用 max_completion_tokens
  "max_completion_tokens": 1024,        // 最大生成 token 数（含 reasoning token）
  "n": 1,                               // 生成几个候选回复（>1 时 usage 按总量计费）

  // ── 惩罚项 ──
  "frequency_penalty": 0.0,             // -2.0~2.0，正值减少重复
  "presence_penalty": 0.0,              // -2.0~2.0，正值鼓励新话题
  "logit_bias": {                       // 强制提高/降低特定 token 的概率
  //  "12345": -100                     // token_id → bias (-100~100)
  },

  // ── Tool Calling ──
  "tools": [                            // 可用工具列表
  //  {
  //    "type": "function",
  //    "function": {
  //      "name": "get_weather",
  //      "description": "获取天气",
  //      "parameters": {
  //        "type": "object",
  //        "properties": {"city": {"type": "string"}},
  //        "required": ["city"]
  //      }
  //    }
  //  }
  ],
  "tool_choice": "auto",                // auto | none | required | {"type":"function","function":{"name":"xxx"}}
  "parallel_tool_calls": true,          // 是否允许并行调多个 tool

  // ⚠️ 以下 function_call / functions 是旧版，已废弃
  // "function_call": "auto",
  // "functions": [...],

  // ── 推理模型 (o-series) ──
  "reasoning_effort": null,             // none | minimal | low | medium | high | xhigh
                                         // 控制推理深度；GPT-5.1 默认 none

  // ── 结构化输出 ──
  "response_format": null,              // null | {"type":"json_object"} | {"type":"json_schema","json_schema":{...}}

  // ── Logprobs ──
  "logprobs": false,                    // 是否返回每个 token 的对数概率
  "top_logprobs": null,                 // 0~20，每个位置返回几个最可能的 token

  // ── 多模态输出 ──
  "modalities": ["text"],               // text | audio 的列表
  "audio": null,                        // {"voice":"alloy", "format":"wav"} 语音输出参数

  // ── Web Search ──
  "web_search_options": null,           // {"search_context_size":"medium","user_location":{...}}

  // ── 内容审核 ──
  "moderation": null,                   // {"model":"omni-moderation-latest"} 对输入输出做审核

  // ── 预测输出 (Predicted Outputs) ──
  "prediction": null,                   // {"content":"预判的静态文本","type":"content"}

  // ── 流式 ──
  "stream": false,                      // true = SSE 流式返回
  "stream_options": null,               // {"include_usage":true} 流式最后一个 chunk 带 usage

  // ── 元数据 & 缓存 ──
  "metadata": null,                     // {"key":"value"} 最多 16 对
  "store": false,                       // 是否存到 OpenAI 用于 distillation/evals
  "prompt_cache_key": "",               // 用户标识，用于缓存优化
  "prompt_cache_retention": "in_memory",// in_memory | 24h

  // ── 安全 ──
  "safety_identifier": "",              // 用于检测滥用（hash 后的用户 ID）
  "user": "",                           // ⚠️ 被 safety_identifier 替代

  // ── 服务等级 ──
  "service_tier": "auto",               // auto | default | flex | scale | priority
  "verbosity": "medium"                 // low | medium | high
}
```

### 2.3 Response JSON —— 全部字段

```json5
{
  // ── 元数据 ──
  "id": "chatcmpl-xxx",                 // 唯一标识
  "object": "chat.completion",          // 永远是 "chat.completion"
  "created": 1781622479,                // Unix timestamp（秒）
  "model": "gemma-4-E4B-it",            // 实际使用的模型
  "system_fingerprint": "fp_xxx",       // 后端配置指纹（与 seed 配合判断确定性）

  // ── 回复内容 ──
  "choices": [
    {
      "index": 0,                       // 候选序号（n>1 时有多个）
      "finish_reason": "stop",          // stop | length | tool_calls | content_filter | function_call(废弃)

      // ◆ message —— 模型生成的消息
      "message": {
        // ── 基础 ──
        "role": "assistant",             // 永远是 "assistant"（响应端）
        "content": "Token 是 LLM 处理文本的最小单位...",  // 文本回复；tool_call 时为 null
        "refusal": null,                 // 安全拒绝原因（如内容违规）

        // ── Tool Calling（当前标准）──
        "tool_calls": null,              // null | [...]；工具调用列表
        // "tool_calls": [{
        //   "id": "call_abc123",        // 唯一 ID，用于关联 tool message
        //   "type": "function",         // 目前只有 function
        //   "function": {
        //     "name": "get_weather",    // 函数名
        //     "arguments": "{\"city\":\"北京\"}"  // JSON string（需 json.loads 解析）
        //   }
        // }],

        // ── Function Call（旧版，已废弃）──
        "function_call": null,           // 旧版单函数调用，用 tool_calls 替代
        // "function_call": {
        //   "name": "get_weather",
        //   "arguments": "{\"city\":\"北京\"}"
        // },

        // ── Web Search 引用标注 ──
        "annotations": null,              // null | [...]；web 搜索时的来源标注
        // "annotations": [{
        //   "type": "url_citation",
        //   "url_citation": {
        //     "url": "https://zh.wikipedia.org/wiki/北京",
        //     "title": "北京 - 维基百科",
        //     "start_index": 8,          // 引用标记在 content 中的起始位置
        //     "end_index": 11            // 引用标记在 content 中的结束位置
        //   }
        // }],

        // ── 音频输出 ──
        "audio": null,                    // null | {id, data, expires_at, transcript}
        // "audio": {
        //   "id": "audio_xxx",
        //   "data": "base64...",         // base64 编码的音频
        //   "expires_at": 1781708879,
        //   "transcript": "语音对应的文字"
        // },

        // ── 推理过程（Gemma 4 / o-series 的 thinking 内容，vLLM 扩展字段）──
        "reasoning": null                 // null | str；模型内部推理的思考过程
        // "reasoning": "我们需要先理解 token 的概念。Token 是文本切分后的最小单位..."
      },

      // ── Logprobs（需请求时 logprobs=true）──
      "logprobs": null                    // null | {content: [{token, logprob, bytes}], refusal: [...]}
    }
  ],

  // ── Token 用量 ──
  "usage": {
    "prompt_tokens": 41,                  // prompt 消耗的 token 数
    "completion_tokens": 52,              // 生成消耗的 token 数
    "total_tokens": 93,                   // prompt + completion
    "prompt_tokens_details": null         // 详情（cache hit tokens 等）
    // "prompt_tokens_details": {
    //   "cached_tokens": 10,             // 命中缓存的 token 数
    //   "audio_tokens": 0,
    //   "text_tokens": 31
    // }
  },

  // ── 服务等级 ──
  "service_tier": null,                   // auto | default | flex | scale | priority

  // ── 审核结果（需请求时 moderation 参数）──
  "moderation": null                     // null | {input: {model, results}, output: {model, results}}

  // ── 以下为 vLLM 专用扩展字段 ──
  // "prompt_logprobs": null,            // prompt 的 logprobs
  // "prompt_token_ids": null,           // prompt 的 token ID 列表
  // "prompt_text": null,                // prompt 的完整文本（含 chat_template 渲染）
  // "kv_transfer_params": null          // KV cache 传输参数（分布式推理用）
}
```

### 2.4 流式 Chunk JSON

```json5
// 第一个 chunk（带 role）
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion.chunk",     // ← 注意是 chunk 不是 completion
  "created": 1781622479,
  "model": "gemma-4-E4B-it",
  "choices": [{
    "index": 0,
    "delta": {                           // ← delta（增量），不是 message
      "role": "assistant",               // 仅第一个 chunk 有
      "content": ""                      // 第一个 chunk content 通常为空
    },
    "finish_reason": null,
    "logprobs": null
  }]
}

// 中间 chunk（逐 token 文本）
{
  "id": "chatcmpl-xxx",                  // 同一个 ID
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "你好"                  // 本次增量文本
      // tool_calls 也可能以增量形式出现在 delta 中（多次累积）
    },
    "finish_reason": null
  }]
}

// 最后一个 chunk（带 finish_reason）
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {},                         // 空 delta
    "finish_reason": "stop"              // ← 结束原因
  }],
  "usage": {                             // stream_options: {include_usage: true} 时出现
    "prompt_tokens": 15,
    "completion_tokens": 9,
    "total_tokens": 24
  }
}
```

### 2.5 流式 SSE 原始数据

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Token"}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" 是"}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" LL"}}]}

...

data: [DONE]
```

---

## 3. 从 messages[] 到 token序列 —— 「拼接」全过程

### 3.1 输入

```json
[
  {"role": "system", "content": "你是助手。"},
  {"role": "user", "content": "你好！"}
]
```

### 3.2 经过 chat_template 渲染

chat_template 是模型 `tokenizer_config.json` 里的 Jinja2 模板，**把结构化的 messages 数组拼成一段连续文本**：

```
<bos><|turn>system
你是助手。<turn|>
<|turn>user
你好！<turn|>
<|turn>model

```

### 3.3 然后 tokenize —— 文本 → 数字

```
token[000] id=    2 → '<bos>'           ← 序列开始
token[001] id=  105 → '<|turn>'         ← turn 开始
token[002] id= 9731 → 'system'          ← role 名
token[003] id=  107 → '\n'              ← 换行
token[004] id=237408 → '你'             ← 正文开始
token[005] id=33813 → '是一个'
...
token[008] id=  106 → '<turn|>'         ← turn 结束
token[009] id=  107 → '\n'
token[010] id=  105 → '<|turn>'         ← 下一个 turn
token[011] id= 2364 → 'user'
token[012] id=  107 → '\n'
token[013] id=144626 → '你好'
token[014] id=  235 → '！'
token[015] id=  106 → '<turn|>'         ← turn 结束
token[016] id=  107 → '\n'
token[017] id=  105 → '<|turn>'         ← add_generation_prompt 加的
token[018] id= 4368 → 'model'           ← 提示模型："该你说话了"
token[019] id=  107 → '\n'
```

### 3.4 Gemma 4 的 token 骨架规则

```
<bos>                              ← id=2,   一句对话永远这样开头
<|turn>ROLE                        ← id=105, 每个 message 的起始
内容...                             ← 文本正文
<turn|>                            ← id=106, 每个 message 的结束
(重复 <|turn>...<turn|>)
<|turn>model\n                     ← add_generation_prompt（提示模型开始生成）
```

**关键点：** `<|turn>` 和 `<turn|>` 是成对的括号。role 名是普通文本 token。LLM 不识别 JSON——它只识别这个 token 序列。chat_template 就是 JSON → token 的翻译器。

---

## 4. Message content 的三种形态

| 形态 | `content` 值 | 何时用 |
|------|-------------|--------|
| 纯文本 | `"你好"` | 99% 常规场景 |
| content blocks | `[{"type":"text","text":"..."}, {"type":"image_url",...}]` | 多模态 |
| null | `None` + `tool_calls=[...]` | assistant 调用工具 |

### 4.1 多模态 message

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "这张图片里有什么？"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
  ]
}
```

支持 type：`text` / `image_url` / `input_audio` / `file`

图片不经过 tokenizer，而是经过 vision encoder 变成 patch embeddings，与文本 embeddings 拼接后送入模型。

---

## 5. Tool Calling —— 核心

### 5.1 完整流程（两轮 HTTP 请求）

```
Round 1                              Round 2
──────                               ──────
POST /v1/chat/completions            POST /v1/chat/completions
{                                     {
  "messages": [                         "messages": [
    {"role":"user",                       {"role":"user",
     "content":"北京天气？"}               "content":"北京天气？"},
  ],                                     {"role":"assistant",
  "tools": [{                             "content":null,
    "type":"function",                     "tool_calls":[{
    "function":{                             "id":"call_abc",
      "name":"get_weather",                  "function":{
      "parameters":{                           "name":"get_weather",
        "properties":{                          "arguments":"{\"city\":\"北京\"}"}
          "city":{"type":"string"}            }}]},
        }                                    {"role":"tool",
      }                                       "tool_call_id":"call_abc",
    }                                         "content":"晴天 25°C"}
  }]                                      ],
}                                       }
        │                                        │
        ▼                                        ▼
{                                       {
  "choices":[{                            "choices":[{
    "finish_reason":"tool_calls",           "finish_reason":"stop",
    "message":{                             "message":{
      "role":"assistant",                     "role":"assistant",
      "content":null,                         "content":"北京今天晴天25°C。"
      "tool_calls":[{                       }
        "id":"call_abc",                  }]
        "function":{                    }
          "name":"get_weather",
          "arguments":"{\"city\":\"北京\"}"
        }
      }]
    }
  }]
}
```

### 5.2 Python 代码实现

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "北京今天天气怎么样？"}]

# ── Round 1：模型决定调用哪个函数 ──
resp1 = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=messages,
    tools=TOOLS,
    max_tokens=200,
)
msg1 = resp1.choices[0].message
# msg1.role         → "assistant"
# msg1.content      → None
# msg1.tool_calls[0].id                 → "chatcmpl-tool-xxx"
# msg1.tool_calls[0].function.name      → "get_weather"
# msg1.tool_calls[0].function.arguments → '{"city":"北京"}'

# ── 你的代码执行工具 ──
import json
args = json.loads(msg1.tool_calls[0].function.arguments)  # {"city":"北京"}
tool_result = get_weather(args["city"])                    # "晴天 25°C"

# ── 把 tool_call 和 tool_result 追加到 messages ──
messages.append({
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": msg1.tool_calls[0].id,
        "type": "function",
        "function": {
            "name": msg1.tool_calls[0].function.name,
            "arguments": msg1.tool_calls[0].function.arguments,
        },
    }],
})
messages.append({
    "role": "tool",
    "tool_call_id": msg1.tool_calls[0].id,  # ← 必须匹配！
    "content": tool_result,                   # ← 工具返回值
})

# ── Round 2：模型基于工具结果给出最终答案 ──
resp2 = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=messages,
    max_tokens=200,
)
# resp2.choices[0].message.content → "北京今天晴天，气温25°C。"
```

### 5.3 tool call 在 token 层面的真实编码（Gemma 4 实测）

纯文本回复：
```
<|turn>model
你好！有什么可以帮你的？<turn|>
```

带 tool_call 的回复（**不是 JSON！**）：
```
<|turn>model
<|tool_call>call:get_weather{city:<|"|>北京<|"|>}<tool_call|>
<turn|>
```

tool result：
```
<|turn>model
<|tool_response>response:get_weather{value:<|"|>晴天25°C<|"|>}<tool_response|>
北京今天晴天。<turn|>
```

**Gemma 4 的 tool call 语法规则：**

```
<|tool_call>                          ← tool call 块开始
call:<函数名>{                         ← 函数名
  <参数名>:<值>,                       ← 参数（字符串值用 <|"|> 包裹！）
  ...
}
<tool_call|>                          ← tool call 块结束

<|tool_response>                      ← tool response 块开始
response:<函数名>{
  value:<|"|>返回值<|"|>
}
<tool_response|>                      ← tool response 块结束
```

**关键发现：**
- `<|"|>` 是特殊的字符串引号 token——**不是普通双引号**
- tool call 和普通文本**可以共存**于同一个 turn
- 不同模型的 tool call 语法完全不同，chat_template 负责双向翻译

### 5.4 tool_call_parser 的双向转换

```
模型原生格式                          OpenAI 标准格式
────────────                          ────────────────
<|tool_call>                          {
call:weather{                           "tool_calls": [{
  city:<|"|>北京<|"|>                       "id": "call_xxx",
}                                           "type": "function",
<tool_call|>                                "function": {
                                              "name": "weather",
              ─── parser 解析 ───►             "arguments": "{\"city\":\"北京\"}"
                                            }
              ◄── template 渲染 ──          }]
                                        }
```

---

## 6. 四种 role 的完整职责

| role | content | 特殊字段 | 何时 |
|------|---------|---------|------|
| `system` | 纯文本 | — | 定义助手行为 |
| `user` | 纯文本 或 content blocks | — | 用户提问 |
| `assistant` | 纯文本 | — | 模型文本回复 |
| `assistant` | **null** | `tool_calls=[...]` | 模型决定调工具 |
| `tool` | 纯文本（工具返回值） | `tool_call_id="call_xxx"` | 工具执行结果 |

---

## 7. 推理引擎收到什么？

推理引擎不关心 messages、JSON、tool_call——它只接收：

```python
EngineCoreRequest(
    prompt_token_ids=[2, 105, 2364, 107, 144626, ...],  # 纯数字序列
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=2048,
        stop=["<turn|>"],     # 遇到这个 token 就停
    ),
)
```

然后逐 token 自回归生成，每步：

```
当前 token 序列 → Embedding → Transformer × L → LM Head → logits
→ 采样（temperature/top-p/top-k）→ 新 token → 拼到末尾 → 继续
```

直到：达到 max_tokens / 输出 EOS / 匹配 stop token。

---

## 8. 一句话总结

```
messages[]                         ← 你写的结构化对话
    │ chat_template (Jinja2)
    ▼
"<bos><|turn>user\n你好<turn|>..." ← 模型原生文本
    │ tokenizer.encode()
    ▼
[2, 105, 2364, 107, 144626, ...]  ← token ID 序列（这是模型真正收到的）
    │ Transformer forward × N
    ▼
[..., 新token, 新token, ...]       ← 逐 token 生成
    │ tokenizer.decode()
    ▼
"你好！我是AI助手..."              ← 返回给你的文本
```

**chat_template 是翻译器，tokenizer 是编码器，推理引擎只认数字。**
