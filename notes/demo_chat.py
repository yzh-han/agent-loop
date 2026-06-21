"""
demo_chat.py —— Chat Completions API 完整学习 Demo

演示从「构造请求」到「收到回复」的完整流程，每一步都有详细的中文注释。
专有名词保留英文：token, prompt, chat template, SSE 等。

用法：
    uv run python src/agent/demo_chat.py

前置条件：
    本地 vLLM 已启动：vllm serve ~/models/google/gemma-4-E4B-it ...
"""

import json
import httpx
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "gemma-4-E4B-it"

# ============================================================
# 准备 messages —— 这是你发给模型的「对话结构」
# ============================================================
# messages 是一个 list，每个元素是一个 dict，包含 role 和 content。
# 模型本身不认识 messages —— 它只认识一串连续的 token。
# messages 会在 vLLM 内部通过 chat_template 被拼接成模型原生格式（见 README 第 5 节）。
MESSAGES = [
    {
        "role": "system",
        "content": "你是一个精炼的助手，回答用中文，不超过3句话。",
    },
    {
        "role": "user",
        "content": "什么是 token？为什么 LLM 不能直接理解文字？",
    },
]

# ============================================================
# Demo 1：非流式请求（non-streaming）—— 等模型全部生成完再返回
# ============================================================
def demo_non_streaming():
    """
    非流式请求：客户端发送请求 → 等待 → 一次性拿到完整回复。

    这是用 openai SDK 的最简写法。
    SDK 内部做的事：
        1. 把 messages 序列化成 JSON
        2. POST 到 http://localhost:8000/v1/chat/completions
        3. 等待 HTTP response body 完整返回
        4. 解析 JSON → ChatCompletion 对象
    """
    print("=" * 60)
    print("Demo 1: 非流式 Chat Completions（用 openai SDK）")
    print("=" * 60)

    # 创建 OpenAI client，指向本地的 vLLM 服务
    client = OpenAI(
        base_url=VLLM_BASE_URL,  # vLLM 的 OpenAI 兼容 endpoint
        api_key="not-needed",    # vLLM 本地部署不需要真实的 API key
    )

    # 调用 Chat Completions API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        temperature=0.7,
        max_tokens=512,
        stream=False,  # 非流式：等全部生成完
    )

    # response 是一个 ChatCompletion 对象（pydantic model）
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))  # 打印原始 JSON
    content = response.choices[0].message.content
    usage = response.usage

    print(f"\n模型回复：\n{content}\n")
    print(f"Token 用量：prompt_tokens={usage.prompt_tokens}, "
          f"completion_tokens={usage.completion_tokens}, "
          f"total_tokens={usage.total_tokens}")
    print()


# ============================================================
# Demo 2：流式请求（streaming）—— 逐 token 收到回复
# ============================================================
def demo_streaming():
    """
    流式请求：客户端发起请求后，vLLM 每生成一个 token 就推送一个 chunk。

    体验：就像 ChatGPT 一样，文字一个字一个字地「蹦出来」。

    底层原理：
        1. HTTP response 的 Content-Type 是 text/event-stream (SSE)
        2. 每生成一个 token，vLLM 就写一行 "data: {...}\n\n"
        3. 生成结束后写 "data: [DONE]"
        4. openai SDK 内部解析 SSE，每个 chunk 封装成 ChatCompletionChunk 对象
    """
    print("=" * 60)
    print("Demo 2: 流式 Chat Completions（用 openai SDK）")
    print("=" * 60)

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    # stream=True：开启流式返回
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        temperature=0.7,
        max_tokens=512,
        stream=True,
    )

    print("\n模型回复（逐 token 流式输出）：")
    collected_text = []
    for chunk in stream:
        # chunk.choices[0].delta.content 是本次推送的新增文本
        # 注意：不是 delta 而是 delta.content，因为可能还有 tool_calls 等其他 delta
        if chunk.choices[0].delta.content:
            delta_text = chunk.choices[0].delta.content
            # print(chunk.model_dump(), end="\n")  # 打印原始 JSON chunk
            print(delta_text, end="", flush=True)
            collected_text.append(delta_text)

    print("\n")
    print(f"完整回复（共 {len(collected_text)} 个 chunk）：{''.join(collected_text)}")
    print()


# ============================================================
# Demo 3：裸 HTTP 请求 —— 用 httpx 直接发起 HTTP 调用
# ============================================================
# 这是为了让您看到「openai SDK 在背后到底发了什么 HTTP 请求」。
# SDK 就是对下面这些 HTTP 调用的封装。
def demo_raw_http():
    """
    用 httpx 直接发送 HTTP 请求，展示底层的 JSON 格式和 SSE 数据流。

    这样你可以清楚地看到：
        1. HTTP request body 长什么样（JSON）
        2. HTTP response headers（尤其是 Content-Type）
        3. SSE 数据流的原始文本
    """
    print("=" * 60)
    print("Demo 3: 裸 HTTP 请求 —— 看清底层数据格式")
    print("=" * 60)

    # ---- 第 1 步：构造 request body（就是 openai SDK 帮你序列化的那个 JSON）----
    request_body = {
        "model": MODEL_NAME,
        "messages": MESSAGES,
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": True,  # 流式，方便看到 SSE 数据流
    }

    print("\n① 发送的 JSON request body：")
    # print(json.dumps(request_body, indent=2, ensure_ascii=False))

    # ---- 第 2 步：发起 HTTP POST 请求 ----
    print("\n② 发起 HTTP POST → http://localhost:8000/v1/chat/completions")
    print("   Headers: Content-Type: application/json")

    client = httpx.Client(timeout=60.0)
    response = client.send(
        client.build_request(
            "POST",
            f"{VLLM_BASE_URL}/chat/completions",
            json=request_body,
            headers={"Content-Type": "application/json"},
        ),
        stream=True,  # httpx 的 stream=True，不一次性读完 body
    )

    print(f"\n③ HTTP 响应状态码：{response.status_code}")
    print(f"   Content-Type: {response.headers.get('content-type')}")

    # ---- 第 3 步：逐行读取 SSE 流 ----
    print("\n④ 逐行读取 SSE 数据流（原始文本）：")
    print("-" * 60)
    line_count = 0
    max_lines = 30  # 只展示前 30 行，避免太长

    for line in response.iter_lines():
        # 每个 SSE chunk 格式: data: {json}\n\n
        # [DONE] 表示流结束
        line_count += 1
        if line_count > max_lines:
            print(f"\n... (省略后续 {line_count} 行，共展示 {max_lines} 行)")
            break
        print(f"  [{line_count:03d}] {line}")

    print("-" * 60)
    print("\n⑤ SSE 格式说明：")
    print("   - 每行以 'data: ' 开头的是有效数据")
    print("   - 空行是 SSE 的事件分隔符")
    print("   - 'data: [DONE]' 表示流结束")
    print("   - openai SDK 帮你解析了这些，封装成 ChatCompletionChunk 对象")
    client.close()


# ============================================================
# Demo 4：模拟 chat template 的效果 —— messages → token IDs
# ============================================================
def demo_tokenization_simulation():
    """
    模拟 vLLM 内部的 tokenization 过程（用请求 tokenizer endpoint）。

    这让你直观看到：
        messages[] → chat_template 拼接 → prompt 文本 → tokenizer.encode → token IDs
    """
    print("=" * 60)
    print("Demo 4: 模拟 tokenization 流程（调用 vLLM tokenizer endpoint）")
    print("=" * 60)

    client = httpx.Client(timeout=30.0)

    # ---- 第 1 步：调用 /tokenize endpoint，看看 token IDs ----
    # 注意：vLLM 的 tokenize endpoint 在 /tokenize（不在 /v1 下）
    resp = client.post(
        "http://localhost:8000/tokenize",
        json={
            "model": MODEL_NAME,
            "messages": MESSAGES,  # vLLM 会先用 chat_template 拼接，再 tokenize
            "add_generation_prompt": True,  # 加上 <start_of_turn>assistant\n
        },
    )
    tokenize_result = resp.json()
    print(json.dumps(tokenize_result, indent=2, ensure_ascii=False))  # 打印原始 JSON
    token_ids = tokenize_result["tokens"]  # 字段名是 "tokens"，是 int list

    print(f"\n① messages 数组的 token 数量：{len(token_ids)}")
    print(f"   前 20 个 token IDs：{token_ids[:20]}")
    print(f"   后 20 个 token IDs：{token_ids[-20:]}")

    # ---- 第 2 步：调用 /detokenize endpoint，看看这些 token 对应的文本 ----
    # 注意：vLLM 的 detokenize endpoint 在 /detokenize（不在 /v1 下）
    resp = client.post(
        "http://localhost:8000/detokenize",
        json={"model": MODEL_NAME, "tokens": token_ids},
    )
    prompt_text = resp.json()["prompt"]

    print(f"\n② 这些 token 对应的完整 prompt 文本（共 {len(prompt_text)} 字符）：")
    print("-" * 60)
    # 把特殊 token 标记出来，方便观察 chat template 的结构
    # 替换不可见的特殊 token 为可见标记
    display_text = (
        prompt_text
        .replace("\n", "↵\n")
    )
    print(display_text)
    print("-" * 60)

    print("\n③ 观察要点：")
    print("   - 注意 messages[] 中的 system 和 user 是如何被拼接的")
    print("   - 注意特殊的 control token（<bos>, <start_of_turn> 等）")
    print("   - 末尾的 '<start_of_turn>assistant\n' 就是 add_generation_prompt")
    print("   - 模型看到这个 prompt 后，就会开始生成 assistant 的回复")

    # ---- 第 3 步：逐个 token 展示映射关系 ----
    print(f"\n④ 逐个 token 的文本映射（前 15 个 + 后 10 个）：")
    for i, tid in enumerate(token_ids[:15]):
        resp = client.post(
            "http://localhost:8000/detokenize",
            json={"model": MODEL_NAME, "tokens": [tid]},
        )
        token_text = resp.json()["prompt"]
        print(f"   token[{i:03d}] id={tid:6d} → 文本='{token_text}'")

    print("   ...")
    for i, tid in enumerate(token_ids[-10:], start=len(token_ids) - 10):
        resp = client.post(
            "http://localhost:8000/detokenize",
            json={"model": MODEL_NAME, "tokens": [tid]},
        )
        token_text = resp.json()["prompt"]
        print(f"   token[{i:03d}] id={tid:6d} → 文本='{token_text}'")

    client.close()
    print()


# ============================================================
# 运行所有 demo
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OpenAI Chat Completions API 学习 Demo")
    print(f"  模型: {MODEL_NAME}")
    print(f"  vLLM: {VLLM_BASE_URL}")
    print("=" * 60 + "\n")

    try:
        # Demo 1: 非流式 —— 最简单的调用方式
        demo_non_streaming()

        # Demo 2: 流式 —— 逐 token 返回
        demo_streaming()

        # Demo 3: 裸 HTTP —— 看清底层数据流
        demo_raw_http()

        # Demo 4: tokenization —— 理解 messages → prompt → token IDs
        demo_tokenization_simulation()

    except httpx.ConnectError:
        print("❌ 无法连接到 vLLM。请确认 vLLM 已启动：")
        print("   curl http://localhost:8000/v1/models")
        print("\n   启动命令示例：")
        print("   vllm serve ~/models/google/gemma-4-E4B-it \\")
        print("     --chat-template-content-format openai \\")
        print("     --max-model-len 65536 \\")
        print("     --served-model-name gemma-4-E4B-it \\")
        print("     --reasoning-parser gemma4 \\")
        print("     --tool-call-parser gemma4")
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
