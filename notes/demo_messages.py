"""
demo_messages.py —— Chat Completions API 深入：Message 的完整形态

核心问题：
    1. 一个 message 里到底能放什么？（text / image_url / tool_call / tool_result / reasoning）
    2. 多模态模型的 message 长什么样？
    3. Agent 模型的 tool calling 接口是怎么工作的？
    4. 这些不同类型的 message 在底层怎么被特殊 token 分割和拼接？

用法：
    uv run python src/agent/demo_messages.py

前置条件：
    vLLM 已启动（Gemma 4 是文本模型，本 demo 主要展示 API 协议层面的格式）
"""

import json
import httpx

# ============================================================
# 配置
# ============================================================
VLLM_BASE_URL = "http://localhost:8000/v1"
TOKENIZE_URL = "http://localhost:8000/tokenize"
DETOKENIZE_URL = "http://localhost:8000/detokenize"
MODEL_NAME = "gemma-4-E4B-it"


# ============================================================
# 辅助函数：tokenize + 展示每个 token 的文本
# ============================================================
def show_tokens(label: str, messages: list, add_generation_prompt: bool = True):
    """
    把 messages 数组 tokenize，展示：
        - prompt 全文（含特殊 token 的可见形式）
        - 每个特殊 token 的位置和含义
        - 各 role 的分割边界
    """
    resp = httpx.post(TOKENIZE_URL, json={
        "model": MODEL_NAME,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    })
    data = resp.json()
    token_ids = data["tokens"]

    # detokenize 整体
    resp2 = httpx.post(DETOKENIZE_URL, json={
        "model": MODEL_NAME,
        "tokens": token_ids,
    })
    prompt_text = resp2.json()["prompt"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  token 总数: {len(token_ids)}")
    print(f"\n  —— prompt 全文（\\n 显示为 ↵）——")
    print(f"  {prompt_text.replace(chr(10), '↵\n  ')}")
    print(f"\n  —— 关键 token 解析 ——")

    # 逐个 token 看
    known_tokens = {
        2: "<bos>",
        105: "<|turn>",
        106: "<turn|>",
        107: "\\n",
    }

    for i, tid in enumerate(token_ids):
        if tid in known_tokens:
            # 看上下文——这个 token 在哪个 role 里
            context_start = max(0, i - 2)
            context_end = min(len(token_ids), i + 3)
            context_ids = token_ids[context_start:context_end]
            resp3 = httpx.post(DETOKENIZE_URL, json={
                "model": MODEL_NAME,
                "tokens": context_ids,
            })
            ctx_text = resp3.json()["prompt"].replace("\n", "↵")
            marker = "◀" if tid in (105, 106) else ""
            print(f"  token[{i:03d}] id={tid:5d} = {known_tokens[tid]:20s}  上下文: \"{ctx_text}\" {marker}")

    # 找出 role 边界
    print(f"\n  —— role 边界 ——")
    prev_text = ""
    for i, tid in enumerate(token_ids):
        resp3 = httpx.post(DETOKENIZE_URL, json={
            "model": MODEL_NAME, "tokens": [tid],
        })
        t = resp3.json()["prompt"]
        if t in ("<|turn>", "<turn|>"):
            context = token_ids[max(0, i - 1): min(len(token_ids), i + 3)]
            resp4 = httpx.post(DETOKENIZE_URL, json={
                "model": MODEL_NAME, "tokens": context,
            })
            ctx = resp4.json()["prompt"].replace("\n", "↵")
            print(f"  token[{i:03d}] id={tid} {t}  周围: \"{ctx}\"")
        prev_text = t

    return token_ids, prompt_text


# ============================================================
# 第 1 部分：Message 的 content 能放什么？
# ============================================================
def part1_message_content_types():
    """
    OpenAI Chat Completions API 中 message.content 的 3 种形态：

    ┌────────────────────┬──────────────────────────────────────────────┐
    │ content 类型        │ 何时使用                                     │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 1. 纯文本 string    │ 最常规的用法，99% 的场景                      │
    │   "你好"            │ "role": "user", "content": "你好"            │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 2. content block   │ 多模态输入（图片/音频/文件）                   │
    │   list[dict]        │ 每个 block 有 type 字段指定媒体类型            │
    ├────────────────────┼──────────────────────────────────────────────┤
    │ 3. null             │ tool_call 消息——只有 tool_calls，没有文字     │
    │                     │ "role": "assistant", "content": null         │
    └────────────────────┴──────────────────────────────────────────────┘
    """

    print("\n" + "#" * 70)
    print("#  第 1 部分：Message content 的三种形态")
    print("#" * 70)

    # ---- 1.1 纯文本 ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 1.1 纯文本 string —— 最常规的形态                                  │
└──────────────────────────────────────────────────────────────────┘""")

    text_only = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "你好！"},
    ]
    show_tokens("纯文本 messages → token 化", text_only)

    # ---- 1.2 多模态 content blocks ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 1.2 content block list —— 多模态输入（图片/音频/文件）              │
│                                                                   │
│ 注意：Gemma 4 是纯文本模型，不支持图片输入。                        │
│ 下面展示的是 OpenAI API 协议支持的标准格式。                        │
│ 当你用 GPT-4o / Gemini 等多模态模型时，就会这样传。                 │
└──────────────────────────────────────────────────────────────────┘""")

    multimodal_example = [
        {
            "role": "user",
            "content": [
                # text block —— 纯文本片段
                {"type": "text", "text": "这张图片里有什么？"},
                # image_url block —— base64 编码的图片
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                        # detail 参数：auto / low / high
                        "detail": "auto",
                    },
                },
            ],
        },
    ]

    print("""
  多模态 message 的 JSON 结构：

  {
    "role": "user",
    "content": [                          # ← content 是 list，不是 string
      {"type": "text", "text": "这张图片里有什么？"},    # 文本块
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}  # 图片块
    ]
  }

  content block 的 type 可以有哪些？

    type="text"         — 纯文本
    type="image_url"    — 图片（HTTP URL 或 base64 data URL）
    type="input_audio"  — 音频（base64 编码的 WAV/PCM）
    type="file"         — 文件（用于文件搜索等）

  每个 block 在 tokenize 时：
    1. 文本块 → 正常 tokenize
    2. 图片块 → 不通过 tokenizer，而是通过 vision encoder 变成 patch embeddings
    3. 拼接：文本 token embeddings + image patch embeddings → 一个混合的 embedding 序列
    4. 特殊 token（如 <image> 占位符）标记图片应该插入的位置
""")

    # ---- 1.3 null content（tool call 消息）----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 1.3 content=null —— assistant 的 tool_call 消息                   │
│                                                                   │
│ 当模型决定调用工具时，它不输出文字，而是输出一个 tool_calls 数组。   │
│ 这时 content 为 null，role 仍然是 assistant。                      │
└──────────────────────────────────────────────────────────────────┘""")

    tool_call_message_example = {
        "role": "assistant",
        "content": None,  # ← null！模型没输出文字，而是输出了 function call
        "tool_calls": [
            {
                "id": "call_abc123",           # tool call 的唯一 ID
                "type": "function",             # 目前只有 function
                "function": {
                    "name": "get_weather",      # 函数名
                    "arguments": '{"city": "北京"}',  # JSON string
                },
            },
        ],
    }

    print(f"""
  assistant tool_call message 的 JSON 结构：
  {json.dumps(tool_call_message_example, indent=6, ensure_ascii=False)}
""")


# ============================================================
# 第 2 部分：Tool Calling 完整流程
# ============================================================
def part2_tool_calling_flow():
    """
    Tool calling 是一个多轮对话循环，每轮都是一个独立的 HTTP 请求。

    流程图：

      ① 用户提问 + tools 定义
         │  POST /v1/chat/completions
         │  {messages: [user_msg], tools: [{...}]}
         ▼
      ② 模型返回 tool_calls（而不是文本）
         │  assistant message: {role: "assistant", content: null, tool_calls: [...]}
         ▼
      ③ 你的代码执行 tool，拿到结果
         │  result = get_weather(city="北京")
         ▼
      ④ 把 tool 执行结果作为 tool message 追加到 messages
         │  POST /v1/chat/completions
         │  {messages: [user_msg, assistant_tool_call_msg, tool_result_msg]}
         ▼
      ⑤ 模型基于 tool 结果生成最终文本回复
         │  assistant message: {role: "assistant", content: "北京今天晴天..."}
         ▼
      ⑥ 如果模型又返回 tool_calls，重复 ③-⑤
    """

    print("\n" + "#" * 70)
    print("#  第 2 部分：Tool Calling 完整流程")
    print("#" * 70)

    # ---- 2.1 定义 tools ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 2.1 tools 定义 —— Function Calling 的 JSON Schema                 │
└──────────────────────────────────────────────────────────────────┘""")

    TOOLS = [
        {
            "type": "function",  # 目前只支持 function
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如 北京、上海",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    print(f"""  tools 的 JSON 定义：
  {json.dumps(TOOLS[0], indent=4, ensure_ascii=False)}

  关键点：
    - tools 是一个数组，可以同时定义多个 function
    - parameters 使用 JSON Schema 格式（与 OpenAI Function Calling 一致）
    - vLLM 会把这个 JSON Schema 嵌入到 chat template 中（用特殊 token 包裹）
    - 模型训练时见过 tool 格式，所以知道如何输出 function call
""")

    # ---- 2.2 完整对话轮次 ----
    print("""\
┌──────────────────────────────────────────────────────────────────┐
│ 2.2 完整的多轮 tool calling messages 序列                         │
└──────────────────────────────────────────────────────────────────┘""")

    full_cycle = [
        # 第 1 轮：用户提问
        {"role": "user", "content": "北京今天天气怎么样？"},
        # 模型返回 tool_call（注意：content=null）
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "北京"}',
                    },
                },
            ],
        },
        # 你的代码执行 tool 后，把结果作为 tool message 追加
        {
            "role": "tool",
            "tool_call_id": "call_abc123",  # 必须和上面的 id 一致
            "content": "北京今天晴天，25°C，湿度40%。",
        },
    ]

    show_tokens("完整 tool calling 循环（第1轮+tool结果）", full_cycle, add_generation_prompt=True)

    # ---- 2.3 各 role 的消息结构 ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 2.3 各 role 在 tool calling 中的职责                              │
└──────────────────────────────────────────────────────────────────┘""")

    roles_table = """
  ┌──────────────┬─────────────────────────────────────────────────┐
  │ role         │ 在 tool calling 中的作用                         │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ system       │ 系统指令（可选），定义助手行为边界                 │
  │ user         │ 用户提问。content 可以是纯文本或 content blocks   │
  │ assistant    │ 两种形态：                                       │
  │              │   a) 普通文本回复：content="你好！"               │
  │              │   b) tool_call：content=null, tool_calls=[...]   │
  │ tool         │ 工具执行结果。必须有 tool_call_id                 │
  │              │ content 是工具返回的字符串结果                     │
  └──────────────┴─────────────────────────────────────────────────┘
"""
    print(roles_table)


# ============================================================
# 第 3 部分：特殊 Token —— 骨架与分割
# ============================================================
def part3_special_tokens():
    """
    这是最核心的部分——理解 chat template 如何用特殊 token 把 messages 数组
    转换成一段连续的、有结构的 token 序列。

    Gemma 4 的特殊 token 体系：

    ┌─────────────┬──────┬────────────────────────────────────────┐
    │ Token 名称   │ ID   │ 含义                                   │
    ├─────────────┼──────┼────────────────────────────────────────┤
    │ <bos>       │ 2    │ Begin of Sequence — 序列开头             │
    │ <|turn>     │ 105  │ 一个新 turn 的开始（非 role 特定）        │
    │ <turn|>     │ 106  │ 一个 turn 的结束                        │
    │ \\n          │ 107  │ 换行符（在结构中起分隔作用）             │
    │ system      │ ...  │ 跟在 <|turn> 后，表示 system turn        │
    │ user        │ ...  │ 跟在 <|turn> 后，表示 user turn          │
    │ model       │ ...  │ 跟在 <|turn> 后，表示 model/assistant turn│
    │ tool IDs    │ ...  │ tool call 的 JSON 被编码为特殊结构        │
    └─────────────┴──────┴────────────────────────────────────────┘

    拼接规则（Gemma 4）：

      <bos>
      <|turn>system\\n{content}<turn|>    ← system prompt
      <|turn>user\\n{content}<turn|>      ← 每个 user message
      <|turn>model\\n{content}<turn|>     ← 每个 assistant message
      <|turn>model\\n{tool_call_json}<turn|>  ← assistant 的 tool call
      <|turn>tool\\n{content}<turn|>      ← 每个 tool result

    注意：
      1. 每个 turn 由 <|turn> 开始，<turn|> 结束——它们是成对的括号
      2. role 名称（system/user/model）是普通文本 token，不是特殊 token
      3. add_generation_prompt=True 时，末尾追加 <|turn>model\\n，提示模型开始生成
    """

    print("\n" + "#" * 70)
    print("#  第 3 部分：特殊 Token —— 骨架与分割")
    print("#" * 70)

    # ---- 3.1 最简单的两轮对话 ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 3.1 简单对话的 token 骨架                                         │
└──────────────────────────────────────────────────────────────────┘""")

    simple = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好！"},
    ]
    show_tokens("简单对话的 token 结构", simple, add_generation_prompt=True)

    # ---- 3.2 带 tool call 的 token 骨架 ----
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 3.2 带 tool calling 的 token 骨架                                 │
└──────────────────────────────────────────────────────────────────┘""")

    # 用更短的工具名，方便看清结构
    tool_cycle = [
        {"role": "user", "content": "北京天气？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"北京"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "晴天25°C"},
    ]
    show_tokens("tool calling 的 token 结构", tool_cycle, add_generation_prompt=True)

    # ---- 3.3 提取规则 ----
    print(r"""
┌──────────────────────────────────────────────────────────────────┐
│ 3.3 如何从 token 流中提取不同部分？                                │
│                                                                   │
│ 从上面的 tokenize 结果我们看到了 Gemma 4 实际的 tool call 格式：    │
└──────────────────────────────────────────────────────────────────┘

  Gemma 4 实际的 tool call token 编码（从 Demo 3.2 实测）：

    纯文本回复的 turn：
      <|turn>model\n你好，有什么可以帮你的？<turn|>

    tool call + 文本混合的 turn：
      <|turn>model\n<|tool_call>call:weather{city:<|"|>北京<|"|>}<tool_call|>\n<|tool_response>response:weather{value:<|"|>晴天25°C<|"|>}<tool_response|>北京今天...<turn|>

  Gemma 4 的 tool call 编码规则：

    ① 整体包裹在 <|turn>model ... <turn|> 内（和普通文本回复一样的外层结构）

    ② 用 <|tool_call> ... <tool_call|> 包裹 tool call 部分
       - 格式：call:<函数名>{<参数名>:<值>, ...}
       - 字符串值用 <|"|> 包裹（不是普通的双引号！这是特殊 token）
       - 示例：call:weather{city:<|"|>北京<|"|>}

    ③ 用 <|tool_response> ... <tool_response|> 包裹 tool 返回结果
       - 格式：response:<函数名>{value:<|"|>返回值<|"|>}
       - 示例：response:weather{value:<|"|>晴天25°C<|"|>}

    ④ tool call 和普通文本可以共存于同一个 turn！
       模型可以先输出 tool_call，然后紧接着输出基于 tool 结果的文本。

  vLLM 的 tool_call_parser（--tool-call-parser gemma4）做的事情：

    生成方向（模型输出 → OpenAI 格式）：
      step 1: 接收原始 token 流，detokenize 成文本
      step 2: 在每个 <|turn>model ... <turn|> 块内，用状态机匹配
              <|tool_call> ... <tool_call|> 和 <|tool_response> ... <tool_response|>
      step 3: 解析 call:func{key:<|"|>val<|"|>} 语法
              → {"name": "func", "arguments": {"key": "val"}}
      step 4: 映射为 OpenAI 格式的 tool_calls：
              {"role": "assistant", "content": null,
               "tool_calls": [{"id": "call_<hash>", "type": "function",
                               "function": {"name": "func",
                                            "arguments": "{\"key\":\"val\"}"}}]}
      step 5: 非 tool_block 的纯文本提取为 content

    输入方向（OpenAI 格式 → 模型原生 token）：
      step 1: 收到 tool_calls 消息
      step 2: chat_template 渲染：
              ① <|turn>model\n
              ② <|tool_call>call:func{arg:<|"|>val<|"|>}<tool_call|>
              ③ 如果有 content 追加文本
              ④ <turn|>
      step 3: tokenize 整个渲染结果 → 发给引擎

  这就是 --tool-call-parser 和 --reasoning-parser 的核心作用：
    - 处理模型原生语法 和 OpenAI 标准格式 之间的双向转换
    - 不同模型的语法完全不同（Gemma 用 <|tool_call|>，LLaMA 用不同的格式）
    - vLLM 用这些参数选择对应模型的 parser
""")


# ============================================================
# 第 4 部分：用裸 HTTP 手动模拟一轮完整的 tool calling
# ============================================================
def part4_raw_tool_calling():
    """
    使用 openai SDK 演示一次完整的 tool calling 循环：

      Round 1:  user 提问 + tools 定义 → assistant 返回 tool_calls
                → 你的代码执行 get_weather() → 拿到结果
      Round 2:  把 tool 结果作为 tool message 追加 → assistant 给出最终答案
    """
    print("\n" + "#" * 70)
    print("#  第 4 部分：真实的 Tool Calling 完整循环（openai SDK）")
    print("#" * 70)

    from openai import OpenAI
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    # ---- 定义 tools ----
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如 北京、上海",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    # ================================================================
    # Round 1：用户提问，模型返回 tool_calls
    # ================================================================
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ Round 1: 用户提问 + tools 定义 → 模型返回 tool_calls               │
└──────────────────────────────────────────────────────────────────┘""")

    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"},
    ]

    print(f"  ▶ 发送 messages: {json.dumps(messages, ensure_ascii=False)}")
    print(f"  ▶ 发送 tools: get_weather(city)")

    response_1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        max_tokens=200,
    )

    msg_1 = response_1.choices[0].message
    print(f"\n  ◀ finish_reason: {response_1.choices[0].finish_reason}")
    print(f"  ◀ message.role: {msg_1.role}")
    print(f"  ◀ message.content: {msg_1.content}")  # 应该是 None
    print(f"  ◀ message.tool_calls:")

    if msg_1.tool_calls:
        for tc in msg_1.tool_calls:
            print(f"      id: {tc.id}")
            print(f"      function.name: {tc.function.name}")
            print(f"      function.arguments: {tc.function.arguments}")
    else:
        print("      (模型没有返回 tool_calls——模型选择直接回答)")
        print(f"      模型文本回复: {msg_1.content}")
        client.close()
        return

    # ================================================================
    # 你的代码执行 tool（模拟）—— 现实中这里调真实 API
    # ================================================================
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ 你的代码执行 tool：get_weather(city="北京")                        │
└──────────────────────────────────────────────────────────────────┘""")

    # 解析 arguments JSON string → dict
    tool_call = msg_1.tool_calls[0]
    import json as _json
    args = _json.loads(tool_call.function.arguments)
    print(f"  arguments 解析: {args}")

    # 模拟执行函数（现实中这里是真实 API 调用）
    MOCK_WEATHER_DB = {
        "北京": "晴天，25°C，湿度40%，空气质量优",
        "上海": "多云，28°C，湿度65%，有微风",
    }
    tool_result = MOCK_WEATHER_DB.get(args.get("city", ""), "暂无数据")
    print(f"  tool 返回结果: {tool_result}")

    # ================================================================
    # Round 2：把 tool call 和 tool result 追加到 messages，模型给出最终答案
    # ================================================================
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ Round 2: 追加 assistant(tool_call) + tool(result) → 最终答案       │
└──────────────────────────────────────────────────────────────────┘""")

    # 把 Round 1 的 assistant message（含 tool_calls）追加到 messages
    messages.append({
        "role": "assistant",
        "content": msg_1.content,  # None
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg_1.tool_calls
        ],
    })

    # 把 tool 执行结果作为 tool message 追加
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,  # 必须匹配上面的 id
        "content": tool_result,
    })

    print(f"  追加后的 messages（共 {len(messages)} 条）：")
    for i, m in enumerate(messages):
        tc_info = ""
        if m.get("tool_calls"):
            tc_info = f" tool_calls[{m['tool_calls'][0]['function']['name']}]"
        if m["role"] == "tool":
            tc_info = f" → call_id={m['tool_call_id'][:20]}..."
        print(f"    [{i}] {m['role']}: content={str(m['content'])[:50]}{tc_info}")

    print(f"\n  ▶ 发送 Round 2 请求...")
    response_2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200,
    )

    msg_2 = response_2.choices[0].message
    print(f"\n  ◀ finish_reason: {response_2.choices[0].finish_reason}")
    print(f"  ◀ 模型最终回答: {msg_2.content}")

    client.close()


# ============================================================
# 第 5 部分：完整 token sequence 总结
# ============================================================
def part5_summary():
    """
    把以上所有内容串起来——展示一个完整对话的 token sequence 全景图。
    """
    print("\n" + "#" * 70)
    print("#  第 5 部分：完整 Token Sequence 全景图")
    print("#" * 70)

    full = [
        {"role": "system", "content": "你是一个天气助手。"},
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        {"role": "user", "content": "北京天气？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city":"北京"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "晴天25°C"},
        {"role": "assistant", "content": "北京今天晴天，25°C。"},
    ]

    ids, text = show_tokens("完整对话全景", full, add_generation_prompt=False)

    # 用注释标注每一段
    print(f"\n  —— 结构逐段标注 ——")
    print(f"""
  <bos>                              ← 序列开始
  <|turn>system
  你是一个天气助手。<turn|>           ← system prompt

  <|turn>user
  你好！<turn|>                      ← 第 1 轮 user

  <|turn>model
  你好！有什么可以帮你的？<turn|>      ← 第 1 轮 assistant 文本回复

  <|turn>user
  北京天气？<turn|>                   ← 第 2 轮 user

  <|turn>model
  <tool_call_json><turn|>           ← 第 2 轮 assistant tool_call（JSON 被编码）
                                        这里没有文字 content，只有结构化的 function call

  <|turn>tool
  晴天25°C<turn|>                    ← tool 执行结果

  <|turn>model
  北京今天晴天，25°C。<turn|>         ← 最终答案（基于 tool 结果）

  关键观察：
    1. 每个 <|turn>...<turn|> 对就是一个 message
    2. role 名称（system/user/model/tool）紧跟在 <|turn> 后面
    3. tool_call 的内容是序列化的 JSON，不是自然语言
    4. add_generation_prompt 会在末尾加 <|turn>model\\n，提示模型开始生成
    5. 模型在生成时看到 <turn|> 就知道 "这个 turn 结束了，该下一个了"
    6. 模型在生成时输出 <turn|> 就是主动说"我说完了"
""")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Chat Completions API 深入学习")
    print("  Message 形态 · Tool Calling · 特殊 Token 骨架")
    print(f"  模型: {MODEL_NAME}")
    print("=" * 70)

    try:
        part1_message_content_types()
        part2_tool_calling_flow()
        part3_special_tokens()
        part4_raw_tool_calling()
        part5_summary()
    except httpx.ConnectError:
        print("\n❌ 无法连接 vLLM。请先启动：vllm serve ...")
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
