"""
run_agent.py —— 启动 Agent

用法：
    # 交互模式（可连续对话）
    uv run python -m agent.run_agent

    # 单次问题
    uv run python -m agent.run_agent --question "北京今天天气怎么样？"

前置条件：
    vLLM 已启动，且开启了 --enable-auto-tool-choice：
    vllm serve ~/models/google/gemma-4-E4B-it \
      --chat-template-content-format openai \
      --max-model-len 65536 \
      --served-model-name gemma-4-E4B-it \
      --reasoning-parser gemma4 \
      --tool-call-parser gemma4 \
      --enable-auto-tool-choice
"""

import sys
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# 从同 package 导入 agent loop 和 tool 定义
from agent.agent_loop import run_agent
from agent.react_loop import run_react
from agent.config import VLLM_BASE_URL, MODEL_NAME
from agent.tool_registry import TOOLS
import agent.tools  # noqa: F401  # 导入 tools 模块以注册工具函数


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个实用的助手，可以使用工具来回答用户问题, 使用工具之前先用plan工具进行计划。"
    "当用户询问天气或需要计算时，请使用对应的工具获取信息。"
    "回答用中文，简洁明了。"
)

INITIAL_INPUT = "先计算187313+3213 = ? 如果是偶数请问今天北京的天气如何? 奇数就问今天上海的天气如何?"

if __name__ == "__main__":
    sys_param = ChatCompletionSystemMessageParam({
        "role": "system",
        "content": SYSTEM_PROMPT,
    })
    user_param = ChatCompletionUserMessageParam({
        "role": "user",
        "content": INITIAL_INPUT,
    })
    final_answer = run_agent(
        sys_param=sys_param,
        user_param=user_param,
        tools=TOOLS,
        max_turns=5,
    )

# # ═══════════════════════════════════════════════════════════════════
# # 单次问答
# # ═══════════════════════════════════════════════════════════════════

# def ask_once(question: str) -> str:
#     """
#     发送一个问题，运行 agent loop，返回最终答案。

#     messages 的初始状态：
#         [system_prompt, user_question]
#     agent loop 可能会在中间插入 assistant(tool_call) 和 tool(result)，
#     最终返回 assistant 的文本回复。
#     """
#     client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

#     # 初始 messages —— agent loop 会原地修改这个 list
#     messages = [
#         SYSTEM_PROMPT,
#         {"role": "user", "content": question},
#     ]

#     print(f"\n{'='*60}")
#     print(f"  用户提问：{question}")
#     print(f"{'='*60}")

#     # 运行 agent loop
#     final_answer = run_agent(
#         client=client,
#         model=MODEL_NAME,
#         messages=messages,
#         tools=TOOLS,
#         max_turns=5,
#         verbose=True,
#     )

#     print(f"\n{'='*60}")
#     print(f"  最终答案：")
#     print(f"  {final_answer}")
#     print(f"{'='*60}\n")

#     return final_answer


# # ═══════════════════════════════════════════════════════════════════
# # 交互模式
# # ═══════════════════════════════════════════════════════════════════

# def interactive():
#     """
#     交互式对话模式。每次用户输入都会运行一次 agent loop。

#     注意：每次提问是独立的 agent run（不共享对话历史）。
#     如果需要多轮对话记忆，可以在循环外维护 messages。
#     """
#     client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

#     print("\n" + "=" * 60)
#     print("  Agent 交互模式")
#     print(f"  模型: {MODEL_NAME}")
#     print("  可用工具: get_weather(city), calculate(expression)")
#     print("  输入 'quit' 或 'exit' 退出")
#     print("=" * 60)

#     while True:
#         try:
#             question = input("\n你: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\n再见！")
#             break

#         if not question:
#             continue
#         if question.lower() in ("quit", "exit", "q"):
#             print("再见！")
#             break

#         messages = [
#             SYSTEM_PROMPT,
#             {"role": "user", "content": question},
#         ]

#         print("Agent: ", end="", flush=True)
#         answer = run_agent(
#             client=client,
#             model=MODEL_NAME,
#             messages=messages,
#             tools=TOOLS,
#             max_turns=5,
#             verbose=False,  # 交互模式少打印中间信息
#         )
#         print(answer)


# # ═══════════════════════════════════════════════════════════════════
# # main
# # ═══════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     # 解析命令行参数（极简版，不依赖 argparse）
#     args = sys.argv[1:]

#     if "--question" in args or "-q" in args:
#         # 单次问答模式
#         idx = args.index("--question") if "--question" in args else args.index("-q")
#         if idx + 1 < len(args):
#             question = args[idx + 1]
#         else:
#             question = input("请输入问题: ")
#         ask_once(question)
#     else:
#         # 默认：交互模式
#         interactive()
