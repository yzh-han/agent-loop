# OpenAI Responses API 学习笔记 —— 从 HTTP 请求到 LLM 推理的全流程

> 本地环境：vLLM (`http://0.0.0.0:8000`) 托管 `google/gemma-4-E4B-it`

---

## 目录

1. [两种 API 对比：Chat Completions vs Responses](#1-两种-api-对比chat-completions-vs-responses)
2. [完整流程图](#2-完整流程图)
3. [第 1 步：构造 HTTP 请求](#3-第-1-步构造-http-请求)
4. [第 2 步：vLLM API Server 收到请求](#4-第-2-步vllm-api-server-收到请求)
5. [第 3 步：Chat Template —— 把 messages 拼接成模型原生格式](#5-第-3-步chat-template--把-messages-拼接成模型原生格式)
6. [第 4 步：Tokenization —— 文本 → token IDs](#6-第-4-步tokenization--文本--token-ids)
7. [第 5 步：Inference Engine —— 推理引擎执行自回归生成](#7-第-5-步inference-engine--推理引擎执行自回归生成)
8. [第 6 步：Detokenization → 流式/非流式返回](#8-第-6-步detokenization--流式非流式返回)
9. [源码分析：vLLM 中 async 推理的完整调用链](#9-源码分析vllm-中-async-推理的完整调用链)
10. [运行 Demo](#10-运行-demo)

---

## 1. 两种 API 对比：Chat Completions vs Responses

| 特性 | `/v1/chat/completions` | `/v1/responses` |
|------|------------------------|-----------------|
| 状态 | ✅ 成熟稳定 | ⚠️ vLLM 支持有限（单轮 OK，多轮有 bug） |
| 输入格式 | `messages: [{role, content}, ...]` | `input: [{type, role, content}, ...]` |
| 输出格式 | `choices[0].message.content` | `output: [{type, role, content}, ...]` |
| 内置 Tools | 需手动处理 tool_calls 循环 | 服务端自动执行 tool 调用 |
| 适用场景 | 通用对话、工具调用 | 多步 agentic 工作流 |

**我们当前用 vLLM 托管 Gemma，优先使用 Chat Completions。** 两个 Demo 都提供，方便对比学习。

---

## 2. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        你的 Python 客户端                            │
│                                                                     │
│   openai.chat.completions.create(                                   │
│     model="gemma-4-E4B-it",                                         │
│     messages=[{"role":"user","content":"你好"}],                     │
│     stream=True                                                     │
│   )                                                                 │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ① HTTP POST /v1/chat/completions
                │    Content-Type: application/json
                │    Body: { "model": "...", "messages": [...], ... }
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   vLLM API Server (FastAPI)                          │
│                     http://0.0.0.0:8000                              │
│                                                                     │
│   ② FastAPI 路由 → api_server.py::create_chat_completion()           │
│      - 解析 JSON body → ChatCompletionRequest (pydantic model)       │
│      - 提取 sampling params（temperature, max_tokens, ...）           │
│      - 提取 messages list                                            │
│      - 调用 chat_template 把 messages 拼接成单个 prompt string        │
│      - 调用 tokenizer.encode(prompt) → token_ids                     │
│      - 构造 SamplingParams + EngineCoreRequest                       │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ③ 发送 EngineCoreRequest 到推理引擎（通过 IPC/multiprocess queue）
                │    request_id, prompt_token_ids, sampling_params
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 vLLM Engine Core (GPU 进程)                          │
│                                                                     │
│   ④ Scheduler 调度：分配 KV cache block，排队等待推理                  │
│   ⑤ Model Runner 执行：                                              │
│      - 将 token_ids 移到 GPU                                        │
│      - 模型 forward pass（Transformer decoder）                      │
│      - 每一步产出 1 个新 token 的 logits                              │
│      - 根据 sampling_params 采样（temperature/top-p/top-k）           │
│      - 采样得到的 token 拼到输入末尾，进入下一步（自回归）              │
│   ⑥ 每生成 1 个 token 或一批 token，通过 queue 发回给 API Server      │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ④ 逐 token 返回（streaming）或全部生成完返回
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   vLLM API Server 收尾                               │
│                                                                     │
│   ⑦ 如果是 streaming：                                              │
│      - 每个 token → detokenize → "data: {...}\n\n" (SSE 格式)        │
│      - yield 给 HTTP 客户端                                          │
│   ⑧ 如果是非流式：                                                   │
│      - 收集所有 token → detokenize → 完整 text                       │
│      - 构造 ChatCompletionResponse → JSON 返回                       │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ⑤ HTTP Response
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        你的 Python 客户端                            │
│                                                                     │
│   for chunk in response:                                            │
│       print(chunk.choices[0].delta.content)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 第 1 步：构造 HTTP 请求

### Chat Completions API 的 JSON Body

```json
{
  "model": "gemma-4-E4B-it",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user",   "content": "用 Python 写一个快速排序。"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

### Responses API 的 JSON Body（对比）

```json
{
  "model": "gemma-4-E4B-it",
  "input": [
    {"type": "message", "role": "system", "content": "你是一个有帮助的助手。"},
    {"type": "message", "role": "user",   "content": "用 Python 写一个快速排序。"}
  ]
}
```

**关键区别：**

- Chat Completions 用 `messages` 字段，每个元素只有 `role` + `content`
- Responses 用 `input` 字段，每个元素有 `type` 字段（`message` | `reasoning` | `function_call` 等），更丰富但也更复杂

---

## 4. 第 2 步：vLLM API Server 收到请求

vLLM 的 API Server 是一个 **FastAPI** 应用。当你 POST 到 `/v1/chat/completions` 时：

```
FastAPI 路由 → 请求体自动校验（Pydantic model）→ 调用 handler 函数
```

### vLLM 收到请求后做的事情（按顺序）：

**Step 2.1 — 解析 & 校验请求**
```
HTTP JSON body
    │
    ▼
ChatCompletionRequest (Pydantic model)
    ├── model:         "gemma-4-E4B-it"      # 模型名 → 查找对应的 vLLM model runner
    ├── messages:      [Message, Message]    # 每个 message 自动校验 role/content
    ├── temperature:   0.7
    ├── max_tokens:    2048
    ├── stream:        true
    └── ...           (top_p, stop, etc.)
```

**Step 2.2 — 提取 messages**
```python
# vLLM 内部大致逻辑（简化版）
for msg in request.messages:
    # msg.role  → "system" | "user" | "assistant" | "tool"
    # msg.content → 文本内容或 multimodal content blocks
```

---

## 5. 第 3 步：Chat Template —— 把 messages 拼接成模型原生格式

这是最容易被忽略但最核心的一步。**LLM 并不理解 `messages` 数组——它只理解一段连续的文本（token sequence）。**

### 什么是 Chat Template？

Chat Template 是一个 **Jinja2 模板**，定义在模型的 `tokenizer_config.json` 中。它把结构化的 `messages` 数组**拼接**成模型训练时使用的**原始文本格式**。

### 具体过程

**输入（messages 数组）：**
```python
[
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user",   "content": "你好！"}
]
```

**经过 chat template 渲染后（Gemma 4 的实际格式，来自 Demo 4 实测）：**
```
<bos><|turn>system
你是一个有帮助的助手。<turn|>
<|turn>user
你好！<turn|>
<|turn>model

```

注意：
- `<bos>` = Begin of Sequence（token id=2，序列开始标记）
- `<|turn>` (token id=105) / `<turn|>` (token id=106) = Gemma 4 的 turn 分隔符
- `model` (token id=4368) — Gemma 4 使用 `model` 而非 `assistant` 作为角色名
- 每个模型的格式不同！LLaMA 用 `[INST]...[/INST]`，GPT 用 `<|im_start|>...<|im_end|>`

### vLLM 中的实际执行

vLLM 启动时加载了 tokenizer 和 chat_template。你的 vLLM 启动参数：
```bash
--chat-template-content-format openai   # 按 OpenAI 的 content 格式来解析
--reasoning-parser gemma4               # 解析 Gemma 4 的 reasoning token
--tool-call-parser gemma4               # 解析 Gemma 4 的 tool call token
```

vLLM 内部调用：
```python
# 伪代码：vLLM 内部的 template 渲染
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # 先不 tokenize，拿到原始文本
    add_generation_prompt=True  # 加上 assistant 的开头标记
)
```

**prompt_text 的结果（Gemma 4）：**
```
<bos><start_of_turn>system
你是一个有帮助的助手。<end_of_turn>
<start_of_turn>user
你好！<end_of_turn>
<|turn>model
```

> **关键洞察：这就是 LLM 真正"看到"的输入。** 模型的训练数据就是这种格式，所以它知道在 `<|turn>model` 之后应该开始生成回复。

---

## 6. 第 4 步：Tokenization —— 文本 → token IDs

拼接好的 prompt_text 还是文本，模型只能处理数字。

### Tokenizer 是什么？

Tokenizer 维护一个 **vocabulary (词表)**，把每个 token 映射到一个整数 ID。

```python
# 伪代码
prompt_text = "<bos><start_of_turn>system\n你是一个有帮助的助手。<end_of_turn>\n..."

token_ids = tokenizer.encode(prompt_text)
# → [2, 106, 1645, 108, 2671, 661, 4005, ...]
#   每个数字对应词表中的一个 token
```

### 不同模型的 tokenizer 不同

```
文本: "你好世界"

Gemma tokenizer:   [1234, 5678, 90]       # 中文可能 1~2 个 token/字
GPT-4 tokenizer:   [5764, 12345]          # 完全不同！
LLaMA tokenizer:   [29871, 123, 456]      # 又不同！
```

**这意味着：同一个 messages 数组，经过不同模型的 chat template + tokenizer，得到的 token sequence 完全不同。**

---

## 7. 第 5 步：Inference Engine —— 推理引擎执行自回归生成

这是 GPU 上真正发生的计算。vLLM 的推理引擎收到：

```
EngineCoreRequest:
  ├── prompt_token_ids: [2, 106, 1645, ..., 108, 2671, ...]   # 共 N 个 token
  ├── sampling_params:
  │     ├── temperature: 0.7
  │     ├── top_p: 1.0
  │     ├── max_tokens: 2048
  │     └── stop: ["<end_of_turn>", ...]
  └── request_id: "cmpl-xxx"
```

### 7.1 Prefill 阶段（处理输入 prompt）

```
prompt_token_ids: [2, 106, 1645, 108, 2671, 661, 4005, ...]
                         │
                         ▼
              ┌─────────────────────┐
              │  Embedding Layer    │  每个 token ID → 稠密向量 (d_model 维)
              │  [N, vocab] → [N, d_model]
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │  Transformer 层 × L │  自注意力 + FFN，并行处理所有 N 个位置
              │  (每层计算 [N, d_model] → [N, d_model])
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │  LM Head            │  最后一个位置的输出 → vocab 上的 logits
              │  [d_model] → [vocab_size]
              └─────────┬───────────┘
                        ▼
                   logits: [vocab_size]  # 每个词的"得分"
```

**关键：Prefill 是并行的。** 输入的所有 N 个 token 一次性处理（利用 Transformer 的自注意力机制），只需一次 forward pass 就得到 "下一个 token 应该是什么" 的信息。

### 7.2 Decode 阶段（逐 token 生成，自回归）

```
Step 1:
  last_token_id = 最后输入的 token
  forward_pass([last_token_id])      # 只处理 1 个 token！
  → logits
  → sample(logits, temperature=0.7)  # 按温度采样
  → new_token_id = 1234              # 采样到的下一个 token

Step 2:
  forward_pass([1234])               # 把新 token 拼到输入末尾
  → logits
  → sample(logits)
  → new_token_id = 5678

Step 3...N:
  重复直到：生成了 max_tokens 个 / 遇到 stop token / 模型输出 EOS

```

**关键：Decode 是串行的。** 每一步只处理 1 个 token（以及之前所有的 KV cache），因为必须知道上一步的输出才能生成下一步。

### 7.3 KV Cache（vLLM 的核心优化）

```
没有 KV Cache：
  Step 100 时，需要重新计算前 100 个 token 的 attention → 浪费

有 KV Cache：
  每个 token 的 Key 和 Value 向量缓存在 GPU 显存中
  Step 100 时，只需要计算第 100 个 token 的 QKV，
  然后和前面 99 个缓存的 KV 做 attention → 极快
```

**vLLM 的 PagedAttention** 把 KV cache 分成固定大小的 "block"，像操作系统的虚拟内存分页一样管理——这是 vLLM 高吞吐量的核心原因。

### 7.4 Sampling（采样策略）

```python
# vLLM 内部的采样逻辑（简化版）
def sample(logits, temperature, top_p, top_k):
    # 1. 温度缩放
    logits = logits / temperature   # temperature=0 时用 greedy

    # 2. Top-K 过滤：只保留概率最高的 K 个 token
    if top_k:
        top_k_indices = topk(logits, top_k)
        logits[~top_k_indices] = -inf

    # 3. Top-P (nucleus) 过滤：保留累积概率 ≤ p 的最小集合
    if top_p < 1.0:
        sorted_logits = sort(logits)
        cumulative_probs = cumsum(softmax(sorted_logits))
        cutoff = cumulative_probs > top_p
        logits[cutoff] = -inf

    # 4. Softmax → 概率分布
    probs = softmax(logits)

    # 5. 按概率采样
    return random.choice(vocab, p=probs)
```

---

## 8. 第 6 步：Detokenization → 流式/非流式返回

### 8.1 推理引擎 → API Server

每生成一个新 token，引擎通过队列发送给 API Server：

```
引擎端（GPU 进程）:
  engine_core.output_queue.put(RequestOutput(
    request_id="cmpl-xxx",
    outputs=[CompletionOutput(
      index=0,
      text="",          # 可能是空的，增量由 token_ids 提供
      token_ids=[1234], # 新生成的 token
    )]
  ))

API Server 端:
  request_output = await output_queue.get()
```

### 8.2 Detokenization

```python
# API Server 把 token ID 变回文字
token_id = 1234
text_chunk = tokenizer.decode([token_id])
# → "你好"（或某个中文字符）
```

### 8.3 Streaming 返回（SSE 格式）

```
HTTP Response:
  Content-Type: text/event-stream

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"你好"},"index":0}]}

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"！"},"index":0}]}

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"我"},"index":0}]}

  ...

  data: [DONE]
```

### 8.4 客户端收到

```python
# openai SDK 内部处理了 SSE 解析，你拿到的是 Python 对象
for chunk in client.chat.completions.create(..., stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 9. 源码分析：vLLM 中 async 推理的完整调用链

> vLLM 版本：>= 0.18.0。下面是 vLLM 源码中的核心路径（简化版，去掉了错误处理、重试等）。

### 文件调用链

```
api_server.py                          # FastAPI 路由
  → openai_serving_chat.py             # Chat Completions handler
    → engine_client.py                 # 异步引擎客户端
      → engine_core.py                 # 推理引擎核心（GPU 进程）
        → model_runner.py              # 实际执行 forward pass
```

### 核心代码片段（简化）

**api_server.py** —— FastAPI 路由定义

```python
@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,  # Pydantic 自动校验
    raw_request: Request,            # FastAPI 原始 request 对象
):
    # 委托给专门 handler
    handler = OpenAIServingChat(
        engine_client=engine_client,
        model_config=model_config,
        ...
    )
    generator = await handler.create_chat_completion(request, raw_request)
    if request.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        return generator
```

**openai_serving_chat.py** —— Chat Completions handler：**这是理解 tokenization 流程最关键的文件**

```python
async def create_chat_completion(self, request, raw_request):
    # ① 应用 chat template —— messages → 文本
    prompt_text = self.tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ② 对 prompt 文本做 tokenize —— 文本 → token IDs
    prompt_token_ids = self.tokenizer.encode(prompt_text)

    # ③ 构造 sampling 参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        ...
    )

    # ④ 发给推理引擎
    async for output in self.engine_client.generate(
        prompt=prompt_token_ids,
        sampling_params=sampling_params,
        request_id=request_id,
        stream=True,
    ):
        # ⑤ 每个新 token → detokenize → SSE chunk
        if output.outputs[0].token_ids:
            delta_text = self.tokenizer.decode(
                output.outputs[0].token_ids[-1]  # 只 decode 最新的一个 token
            )
            yield ChatCompletionChunk(
                choices=[Choice(delta=Delta(content=delta_text))],
                ...
            )
```

**engine_core.py** —— 推理引擎的核心循环（GPU 进程内部）

```python
class EngineCore:
    def step(self):
        # ① Scheduler 从等待队列中选出可以执行的请求
        scheduled = self.scheduler.schedule()

        # ② 把 token_ids 打包成 GPU 可以执行的 batch
        model_input = self.prepare_input(scheduled)
        # model_input: {input_ids: Tensor[batch, seq_len],
        #               positions: Tensor[batch, seq_len]}

        # ③ 模型 forward pass
        hidden_states = self.model(model_input)

        # ④ 采样
        sampled_tokens = self.sampler(
            hidden_states,
            sampling_params,  # temperature, top_p 等
        )

        # ⑤ 把新 token 发给 API Server（通过 IPC queue）
        for req_id, token in zip(scheduled.request_ids, sampled_tokens):
            self.output_queue.put(RequestOutput(
                request_id=req_id,
                token_ids=[token],
            ))
```

### 一图总结调用链

```
客户端 openai SDK call
    │
    ▼
vLLM FastAPI (api_server.py)
    │ create_chat_completion()
    ▼
OpenAIServingChat (openai_serving_chat.py)
    │ apply_chat_template(messages) → prompt_text
    │ tokenizer.encode(prompt_text) → token_ids
    │ SamplingParams(...)
    ▼
AsyncEngineClient (engine_client.py)
    │ add_request(prompt_token_ids, sampling_params)
    │ IPC / multiprocess queue
    ▼
EngineCore (engine_core.py) — GPU 进程
    │ scheduler.schedule()
    │ model_runner.execute_model()
    │ sampler.sample()
    │ output_queue.put(new_token)
    ▼
[IPC queue 传回 API Server]
    │ tokenizer.decode(new_token) → text
    │ StreamingResponse (SSE)
    ▼
客户端收到 chunk
```

---

## 10. 运行 Demo

### 前置条件

```bash
# 安装依赖
uv sync

# 确认 vLLM 在运行
curl http://localhost:8000/v1/models
```

### 运行 Chat Completions Demo

```bash
uv run python src/agent/demo_chat.py
```

### 运行 Responses API Demo（实验性）

```bash
uv run python src/agent/demo_responses.py
```

---

## 11. 深入：Message 的完整形态、Tool Calling 与特殊 Token 骨架

> 本节的完整可运行 Demo：[demo_messages.py](src/agent/demo_messages.py)

### 11.1 Message content 的三种形态

Chat Completions API 中 `message.content` 可以是：

| 形态 | Python 类型 | 示例 | 何时使用 |
|------|-----------|------|---------|
| **纯文本** | `str` | `"你好"` | 99% 的常规场景 |
| **content blocks** | `list[dict]` | `[{"type":"text","text":"..."}, {"type":"image_url","image_url":{...}}]` | 多模态输入 |
| **null** | `None` | `content=None, tool_calls=[...]` | assistant 的 tool_call 消息 |

**多模态 content block 的类型：**

```python
# 图片 + 文本混合输入
{
    "role": "user",
    "content": [
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."  # 或 HTTPS URL
        }},
    ]
}
```

支持的 `type`：`text` / `image_url` / `input_audio` / `file`

### 11.2 Tool Calling 完整流程

```
第 1 次 HTTP 请求                  第 2 次 HTTP 请求
┌──────────────────────┐          ┌──────────────────────────────┐
│ messages:            │          │ messages:                    │
│   [user: "天气?"]    │          │   [user: "天气?"]            │
│ tools: [get_weather] │          │   [assistant: tool_call]     │
└──────────┬───────────┘          │   [tool: "晴天 25°C"]  ← 新! │
           │                      └──────────────┬───────────────┘
           ▼                                     │
┌──────────────────────┐                         ▼
│ 模型返回:             │          ┌──────────────────────────────┐
│   role: assistant    │          │ 模型返回:                     │
│   content: null      │          │   role: assistant            │
│   tool_calls: [{     │          │   content: "北京今天晴天..."   │
│     function: {      │          └──────────────────────────────┘
│       name:"get_weather",
│       arguments:"..." │
}}]                     │
└──────────────────────┘
```

### 11.3 Gemma 4 的特殊 Token 骨架（实测）

调用 vLLM 的 `/tokenize` endpoint 揭开了模型实际看到的 token sequence：

```
<bos>                                   ← id=2,  序列开始
<|turn>system                            ← id=105, turn 开始
你是一个天气助手。 <turn|>               ← id=106, turn 结束
<|turn>user
你好！<turn|>
<|turn>model
你好！有什么可以帮你的？<turn|>
<|turn>user
北京天气？<turn|>
<|turn>model
<|tool_call>call:weather{city:<|"|>北京<|"|>}<tool_call|>
<|tool_response>response:weather{value:<|"|>晴天25°C<|"|>}<tool_response|>
北京今天晴天。<turn|>
```

**关键发现：**

1. `<|turn>` (id=105) 和 `<turn|>` (id=106) 是**成对的括号**，包裹每一个对话轮次
2. role 名 (`system`/`user`/`model`/`tool`) 是普通文本 token，紧跟在 `<|turn>` 后
3. tool call 用 `<|tool_call>call:函数名{参数:<|"|>值<|"|>}<tool_call|>` 编码——**不是 JSON！**
4. 字符串值用特殊 token `<|"|>` 包裹，而不是普通的双引号
5. tool response 用 `<|tool_response>response:函数名{value:<|"|>结果<|"|>}<tool_response|>` 包裹
6. **tool call 可以和普通文本共存于同一个 turn**（模型可以同时输出 tool call 和总结文字）
7. `add_generation_prompt=True` 时末尾追加 `<|turn>model\n`——这告诉模型"该你说话了"

### 11.4 tool_call_parser 的双向转换

```
                    ┌─────────────────────┐
                    │  tool_call_parser   │
                    │  (gemma4)           │
                    └─────────────────────┘
                           │
        生成方向              │          输入方向
        (模型→API)           │          (API→模型)
                           │
  <|tool_call>              │    tool_calls: [{
    call:weather{           │      function: {
      city:<|"|>北京<|"|>   │        name: "weather",
    }                       │        arguments: '{"city":"北京"}'
  <tool_call|>              │      }
  ──────────────────────────│──→  }]
         解析               │            渲染
                            │   ─────────────────
          │                 │            │
          ▼                 │            ▼
  {"name":"weather",        │   <|tool_call>
   "arguments":             │     call:weather{
     {"city":"北京"}}       │       city:<|"|>北京<|"|>
                            │     }
                            │   <tool_call|>
```

### 11.5 不同模型的不同 Token 体系

| 模型族 | Turn 分隔 | Tool call 格式 | 信源 |
|--------|----------|---------------|------|
| **Gemma 4** | `<|turn>` / `<turn|>` | `<|tool_call>call:fn{...}<tool_call|>` | 本文实测 |
| **LLaMA 3** | `<|start_header_id|>` / `<|eot_id|>` | JSON 嵌入在 `<|python_tag|>` 中 | Meta 文档 |
| **GPT-4** | `<|im_start|>` / `<|im_end|>` | 函数调用放在顶层 JSON（API 不暴露 token） | OpenAI 文档 |
| **DeepSeek** | `</s>` 分隔 + role 前缀 | JSON 嵌入在 `則` / `${` 标记中 | DeepSeek 论文 |

**核心认知：chat template 就是模型的"语言翻译器"。** 它把统一的 OpenAI messages 格式翻译成每个模型独有的 token 方言。你写的代码不需要关心这些差异——但理解它们能让你在调试时不再抓瞎。

---

## 补充：关键概念速查

| 概念 | 含义 | 在哪一步 |
|------|------|---------|
| **Chat Template** | Jinja2 模板，把 `messages[]` 渲染成模型原生格式的文本 | 步骤 3 |
| **Tokenizer / encode** | 文本 → token IDs | 步骤 4 |
| **Tokenizer / decode** | token IDs → 文本 | 步骤 6 |
| **Prefill** | 并行处理输入 prompt 的所有 token | 步骤 5.1 |
| **Decode** | 逐 token 自回归生成 | 步骤 5.2 |
| **KV Cache** | 缓存已计算 token 的 Key/Value 向量，避免重复计算 | 步骤 5.3 |
| **PagedAttention** | vLLM 的分页式 KV cache 管理 | 步骤 5.3 |
| **SamplingParams** | 控制采样行为：temperature, top_p, top_k 等 | 步骤 5.4 |
| **SSE** | Server-Sent Events，流式返回的 HTTP 协议 | 步骤 6.3 |
| **EOS** | End of Sequence token，模型表示"我生成完了" | 步骤 5.2 |

---

## 参考资料

- [vLLM OpenAI-Compatible Server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [OpenAI Chat Completions API 文档](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Responses API 文档](https://platform.openai.com/docs/api-reference/responses)
- [vLLM Responses API RFC](https://github.com/vllm-project/vllm/issues/32850)
