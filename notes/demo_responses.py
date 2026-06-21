"""
demo_responses.py —— OpenAI Responses API 学习 Demo

OpenAI 在 2024 年底推出了全新的 Responses API (/v1/responses)。
它与 Chat Completions API 的核心区别：

    Chat Completions:
        - 无状态：每次请求都要携带完整 messages 历史
        - 输出简单：choices[0].message.content
        - Tool calling 需要手动循环

    Responses API:
        - 有状态：服务端保存 conversation state
        - 输出丰富：output 包含 message/reasoning/function_call 等多种 item type
        - 内置 tool 执行：web_search, file_search, code_interpreter 等

⚠️ vLLM 目前的 /v1/responses 支持有限：
    - 单轮 OK，多轮有 bug（详见 README 第 1 节）
    - 本 Demo 用于学习 API 格式，不推荐生产使用

用法：
    uv run python src/agent/demo_responses.py
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
# Demo 1：用 openai SDK 调用 Responses API
# ============================================================
def demo_responses_sdk():
    """
    使用 openai SDK 的 client.responses.create() 方法。

    注意 input 的格式：
        - 每个 item 有 type 字段（"message" | "reasoning" | "function_call" 等）
        - 而不是 Chat API 的 messages 里的 role + content

    注意 output 的格式：
        - response.output 是一个 list
        - 每个 item 也有 type 字段
        - 文本回复通常 type="message"
    """
    print("=" * 60)
    print("Demo 1: Responses API（用 openai SDK）")
    print("=" * 60)

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    # ======== 关键区别：input 字段 vs Chat API 的 messages 字段 ========
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            # Responses API 用 input，不用 messages
            # 每个 item 必须指定 type 字段
            input=[
                {
                    "type": "message",     # ← 必须有 type
                    "role": "system",
                    "content": "你是一个精炼的助手，回答用中文。",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "什么是 token？",
                },
            ],
            # 注意：Responses API 的参数名也不同
            # max_output_tokens 而不是 max_tokens（Chat API 的参数）
            max_output_tokens=512,
            temperature=0.7,
        )

        print(f"\n响应对象类型：{type(response).__name__}")
        print(f"Response ID: {response.id}")
        print(f"Status: {response.status}")

        # ======== 输出格式：output list vs Chat API 的 choices list ========
        print(f"\noutput 有 {len(response.output)} 个 item：")
        for i, item in enumerate(response.output):
            print(f"  [{i}] type={item.type}, role={item.role}")
            if hasattr(item, "content"):
                # content 可能是 str 或 list[ContentBlock]
                for j, block in enumerate(item.content):
                    if hasattr(block, "text"):
                        print(f"      content[{j}] text: {block.text[:200]}")
            # Responses API 可能包含 reasoning item（思考过程）
            if hasattr(item, "summary"):
                print(f"      summary: {item.summary[:200]}")

        if response.usage:
            print(f"\nToken 用量：input={response.usage.input_tokens}, "
                  f"output={response.usage.output_tokens}")

    except Exception as e:
        print(f"\n⚠️ Responses API 调用失败（vLLM 支持尚不完善）：")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n   这是预期的行为。vLLM 的 /v1/responses 目前单轮可用，")
        print(f"   但某些响应格式可能不兼容。")
        print(f"   生产环境建议使用 /v1/chat/completions。")


# ============================================================
# Demo 2：裸 HTTP 对比 —— Chat API vs Responses API 的 JSON 格式
# ============================================================
def demo_format_comparison():
    """
    直接对比两种 API 的 HTTP request body 格式差异。

    这让你清楚地看到：
        - Chat API 的 request body 长什么样
        - Responses API 的 request body 长什么样
        - 两者的核心区别在哪里
    """
    print("\n" + "=" * 60)
    print("Demo 2: 两种 API 的 JSON 格式对比")
    print("=" * 60)

    # ---- Chat Completions 的 JSON ----
    chat_body = {
        "model": MODEL_NAME,
        "messages": [  # ← 用 messages
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "什么是 token？"},
        ],
        "temperature": 0.7,
        "max_tokens": 512,  # ← 用 max_tokens
        "stream": False,
    }

    print("\n① Chat Completions API (/v1/chat/completions) 的 request body：")
    print(json.dumps(chat_body, indent=2, ensure_ascii=False))

    # ---- Responses API 的 JSON ----
    responses_body = {
        "model": MODEL_NAME,
        "input": [  # ← 用 input（不是 messages）
            {"type": "message", "role": "system", "content": "你是一个助手。"},
            # ↑ 必须有 type 字段
            {"type": "message", "role": "user", "content": "什么是 token？"},
        ],
        "temperature": 0.7,
        "max_output_tokens": 512,  # ← 用 max_output_tokens（不是 max_tokens）
        "stream": False,
    }

    print("\n② Responses API (/v1/responses) 的 request body：")
    print(json.dumps(responses_body, indent=2, ensure_ascii=False))

    # ---- 关键区别总结 ----
    print("\n③ 关键区别：")
    differences = [
        ("顶层字段", "messages", "input",
         "Chat 用 messages，Responses 用 input"),
        ("每个元素", '{"role": "...", "content": "..."}',
         '{"type": "message", "role": "...", "content": "..."}',
         "Responses 每个 item 多了 type 字段，支持 message/reasoning/function_call 等类型"),
        ("max tokens 参数", "max_tokens", "max_output_tokens",
         "参数名不同"),
        ("输出格式", "choices[0].message.content", "output[0].content[*].text",
         "输出结构完全不同，Responses 的输出层级更深，类型更丰富"),
        ("状态管理", "无状态（每次携带全量 messages）", "有状态（服务端保存 conversation）",
         "Responses 通过 previous_response_id 可以引用之前的对话"),
    ]

    for aspect, chat, resp, note in differences:
        print(f"   {aspect}:")
        print(f"     Chat API:       {chat}")
        print(f"     Responses API:  {resp}")
        print(f"     → {note}")
        print()


# ============================================================
# Demo 3：用 raw HTTP 直接调 Responses API
# ============================================================
def demo_responses_raw_http():
    """
    用 httpx 直接 POST 到 /v1/responses，查看原始响应。
    """
    print("=" * 60)
    print("Demo 3: 用裸 HTTP 调用 Responses API")
    print("=" * 60)

    request_body = {
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "你好！"}
        ],
        "max_output_tokens": 200,
        "temperature": 0.7,
    }

    print("\n① 发送的 JSON request body：")
    print(json.dumps(request_body, indent=2, ensure_ascii=False))

    print(f"\n② 发起 HTTP POST → {VLLM_BASE_URL}/responses")

    client = httpx.Client(timeout=60.0)

    try:
        response = client.post(
            f"{VLLM_BASE_URL}/responses",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        print(f"\n③ HTTP 响应状态码：{response.status_code}")

        if response.status_code == 200:
            body = response.json()
            print(f"\n④ 响应 JSON（格式化）：")
            # 简化 output 的展示
            for item in body.get("output", []):
                if isinstance(item, dict) and "content" in item:
                    for c in item["content"]:
                        if isinstance(c, dict) and "text" in c:
                            c["text"] = c["text"][:300] + "..." if len(c.get("text", "")) > 300 else c.get("text", "")
            print(json.dumps(body, indent=2, ensure_ascii=False))
        else:
            print(f"\n④ 错误响应：")
            print(response.text[:1000])

    except httpx.ConnectError:
        print("❌ 无法连接到 vLLM")
    except Exception as e:
        print(f"⚠️ 错误: {type(e).__name__}: {e}")
    finally:
        client.close()

    print()


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OpenAI Responses API 学习 Demo")
    print(f"  模型: {MODEL_NAME}")
    print(f"  vLLM: {VLLM_BASE_URL}")
    print("=" * 60)
    print()
    print("⚠️  注意：vLLM 对 /v1/responses 的支持尚不完善。")
    print("   单轮对话通常可以工作，但多轮和 tool calling 有已知 bug。")
    print("   本 Demo 主要用于理解 API 格式差异，生产请用 /v1/chat/completions。")
    print()

    try:
        # Demo 1: SDK 调用 Responses API
        demo_responses_sdk()

        # Demo 2: 格式对比
        demo_format_comparison()

        # Demo 3: 裸 HTTP
        demo_responses_raw_http()

    except httpx.ConnectError:
        print("❌ 无法连接到 vLLM。请确认 vLLM 已启动：")
        print("   curl http://localhost:8000/v1/models")
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
