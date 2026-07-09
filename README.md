# OpenAI Compatible Agent Loop 学习笔记

这个仓库实现了一个最小可运行的 OpenAI Chat Completions 兼容 agent loop：

- 使用本地 vLLM 提供的 OpenAI-compatible `/v1/chat/completions`
- 通过 OpenAI Python SDK 调用模型
- 支持 function tool calling
- 自动把 Python 函数注册成 OpenAI tools JSON Schema
- 执行工具后把 `assistant(tool_calls)` 和 `tool` 结果追加回 messages
- 最后调用 vLLM `/tokenize` 和 `/detokenize` 打印 chat template/token mapping，方便观察模型实际看到的 prompt

当前代码默认面向本地 vLLM 托管的 `gemma-4-E4B-it`。

## How To Run

### 1. 准备 `.env`

```bash
cp .env_example .env
```

默认配置见 [.env_example](/home/yzh/code/agent/.env_example)：

```dotenv
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_TOKENIZE_URL=http://localhost:8000/tokenize
VLLM_DETOKENIZE_URL=http://localhost:8000/detokenize
MODEL_NAME=gemma-4-E4B-it
TEMPERATURE=0.7
MAX_TOKENS=1024
AGENT_MAX_TURNS=5
```

[config.py](/home/yzh/code/agent/src/agent/config.py) 还会读取可选的 `API_KEY`。本地 vLLM 通常不校验 key，可以不填；如果你的服务要求鉴权，就在 `.env` 里补上：

```dotenv
API_KEY=not-needed
```

### 2. 启动 vLLM (可以跳过, 已经配置好了见飞书)

代码注释中使用的启动方式如下：

```bash
vllm serve ~/models/google/gemma-4-E4B-it \
  --chat-template-content-format openai \
  --max-model-len 65536 \
  --served-model-name gemma-4-E4B-it \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice
```

可以先确认服务可访问：

```bash
curl http://localhost:8000/v1/models
```

### 3. 安装依赖并运行

```bash
uv sync
uv run python -m agent.run_agent
```

注意：当前 [run_agent.py](/home/yzh/code/agent/src/agent/run_agent.py) 没有实现交互模式，也没有解析 `--question` 参数。它会直接运行代码里的固定问题：

```python
INITIAL_INPUT = "先计算187313+3213 = ? 如果是偶数请问今天北京的天气如何? 奇数就问今天上海的天气如何?"
```

如果要修改问题，直接改 `INITIAL_INPUT`。

## 当前代码结构

```text
src/agent/
  run_agent.py       # 程序入口：构造 system/user message，调用 agent loop
  agent_loop.py      # 核心 agent loop：调用模型、处理 tool_calls、执行工具、返回最终答案
  tool_registry.py   # 工具注册框架：函数签名 -> OpenAI tools JSON Schema
  tools.py           # 具体工具函数：get_weather / calculate / plan
  config.py          # 从 .env 加载 vLLM URL、模型名、API key 等配置
  utils.py           # 调用 /tokenize 和 /detokenize，打印 token mapping

notes/
  demo_chat.py
  demo_messages.py
  demo_responses.py
  NOTES.md
```

## Agent Loop 实际流程

[agent_loop.py](/home/yzh/code/agent/src/agent/agent_loop.py) 的核心逻辑是一个循环：

```text
system/user messages
  -> chat.completions.create(...)
  -> 如果模型返回 tool_calls：
       1. 把 assistant tool_calls 追加进 messages
       2. 调用 execute_tool(...)
       3. 把 tool result 追加进 messages
       4. 进入下一轮 LLM 调用
  -> 如果模型返回最终 content：
       1. 追加 assistant content
       2. 打印 token mapping
       3. return 最终答案
```

当前每次 LLM 调用会传：

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": True}
}
```

所以日志里会打印：

- `[thinking]`：模型 reasoning 字段
- `[finish_reason]`：本轮结束原因
- `[usage]`：token 使用量
- tool calling 详情
- 最终答案预览
- chat template/token mapping

## 当前可用工具

工具定义在 [tools.py](/home/yzh/code/agent/src/agent/tools.py)，导入该模块后会通过 `@register_tool()` 自动注册。

### `calculate`

计算数学表达式，支持加减乘除和括号。

参数：

```python
expression: str
```

实现上使用：

```python
eval(expression, {"__builtins__": {}}, {})
```

这是学习 demo 写法，不建议直接用于不可信输入的生产环境。

### `get_weather`

查询内置天气数据。

参数：

```python
city: Literal["北京", "上海", "深圳", "杭州", "成都"]
```

天气数据是本地 hard-coded 示例，不会访问真实天气 API。

## 如何添加新工具

在 [tools.py](/home/yzh/code/agent/src/agent/tools.py) 里新增函数并加上 `@register_tool()`：

```python
from typing import Annotated
from agent.tool_registry import register_tool

@register_tool()
def echo(text: Annotated[str, "要原样返回的文本"]) -> str:
    """原样返回输入文本。"""
    return text
```

[tool_registry.py](/home/yzh/code/agent/src/agent/tool_registry.py) 会根据函数签名和 docstring 自动生成 OpenAI tool schema，并把函数加入执行注册表。

支持的参数注解：

- `str` / `int` / `float` / `bool`
- `Annotated[T, "描述"]`
- `Literal[...]`
- `Optional[T]` 或 `T | None`
- `list[T]`
- `dict[str, T]`

不支持的注解会在注册阶段直接抛错。

## Chat Completions 与 Responses

当前主代码使用 `/v1/chat/completions`，不是 `/v1/responses`。

| 特性 | `/v1/chat/completions` | `/v1/responses` |
|------|------------------------|-----------------|
| 当前主流程 | 是 | 否 |
| 输入格式 | `messages: [{role, content}, ...]` | `input: [...]` |
| tool calling | 客户端手动执行 tool loop | 服务端/客户端能力依实现而定 |
| 本仓库用途 | agent loop 主路径 | 仅在 `notes/` 中作为学习对比 |

## Message 与 Tool Calling 形态

Chat Completions 中常见 message 形态：

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": None, "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
{"role": "assistant", "content": "最终答案"}
```

一次工具调用通常是两轮请求：

```text
第 1 轮：
  user + tools -> 模型返回 assistant.tool_calls

本地执行：
  execute_tool(name, arguments) -> tool result

第 2 轮：
  user + assistant.tool_calls + tool result + tools -> 模型返回最终答案
```

## Token Mapping 调试

最终答案生成后，[utils.py](/home/yzh/code/agent/src/agent/utils.py) 会调用：

- `POST /tokenize`
- `POST /detokenize`

目的是把当前 messages 和 tools 经 vLLM chat template 渲染后的文本打印出来，观察 Gemma 工具调用在 token 层的实际骨架，例如 turn、tool declaration、tool call、tool response 等特殊标记。

这部分依赖 vLLM 暴露 `/tokenize` 和 `/detokenize` endpoint。如果你的 vLLM 服务没有开启或 URL 配置不对，模型调用可能成功，但 token mapping 打印会失败。

## 目前限制

- `run_agent.py` 当前只有固定问题 demo，没有交互式 CLI。
- `.env` 中的 `TEMPERATURE`、`MAX_TOKENS`、`AGENT_MAX_TURNS` 已被配置类读取，但当前主流程没有全部使用：`run_agent.py` 直接传 `max_turns=5`，`agent_loop.run_agent()` 默认 `max_tokens=8192`。
- `calculate` 是 demo 工具，不适合作为生产级表达式求值器。
- `get_weather` 使用本地假数据，不查询实时天气。
- `pyproject.toml` 中声明的 `agent = "agent:main"` 脚本当前没有对应 `main` 函数，建议使用 `uv run python -m agent.run_agent`。

## 参考资料

- [vLLM OpenAI-Compatible Server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [OpenAI Chat Completions API 文档](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Responses API 文档](https://platform.openai.com/docs/api-reference/responses)
