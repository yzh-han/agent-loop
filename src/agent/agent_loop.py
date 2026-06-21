"""
agent_loop.py —— 最小 Agent Loop

════════════════════════════════════════════════════════════════════
核心: 一个 while 循环, 在"LLM 推理"和"工具执行"之间来回, 直到 LLM 给出最终文本.

循环图示:
  msg_params ──► LLM ──► 返回 tool_calls?
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           tool_calls       content       error
           is not None      is not None
                │             │             │
                ▼             ▼             ▼
         执行工具函数      最终文本回复    报错退出
                │           return
                ▼
         追加 assistant(tool_calls) + tool(result) 到 msg_params
                │
                └──► 回到循环开头 (继续调 LLM)

════════════════════════════════════════════════════════════════════
数据流 —— msg_params 的 role 序列演变:

  初始化:
    ChatCompletionSystemMessageParam      ← 0 或 1 条, 总是在最前面
    ChatCompletionUserMessageParam        ← 用户提问

  Round 1 → LLM 返回 tool_calls →
    ChatCompletionAssistantMessageParam   ← content=null, tool_calls=[get_weather, calculate]
        (tool_calls 是 ChatCompletionMessageToolCallUnionParam 列表, 可并行多个)
    ChatCompletionToolMessageParam        ← role="tool", tool_call_id="call_1", content="晴天"
    ChatCompletionToolMessageParam        ← role="tool", tool_call_id="call_2", content="58.3"

  Round 2 → LLM 返回文本 →
    ChatCompletionAssistantMessageParam   ← content="北京晴天25°C, 计算结果58.3"
        (tool_calls 不出现或为空列表, loop 结束)

  如果模型继续调工具, Assistant(tool_calls) + Tool 可以重复多轮.

════════════════════════════════════════════════════════════════════
tool_call_id 的作用:

  token 层:
    - 没有 tool_call_id 这个概念.
    - 模型靠 <|tool_call> 和 <|tool_response> 的出现顺序一一对应.
    - 第 1 个 <|tool_call> 的输出对应第 1 个 <|tool_response> 的结果, 以此类推.

  API 层:
    - tool_call_id 用于在拼 messages 时建立关联.
    - chat_template 靠它按 tool_calls 数组顺序排列 tool 结果.
    - 即使你把 tool message 顺序写反了, chat_template 也会纠正.
    - 所以它本质上是一个冗余的匹配字段——顺序本身已经足够.

════════════════════════════════════════════════════════════════════
每轮都传 tools 参数:

  每轮都传, 不用额外处理. tools 定义被 chat_template 渲染为:

    <|tool>declaration:func_name{description:<|"|>描述<|"|>,parameters:{...}}<tool|>

  放在 system turn 中. vLLM 的 prefix cache 会缓存这部分, 后续轮次不重复计算 token.

════════════════════════════════════════════════════════════════════
response 中哪些字段拼回 msg_params, 哪些不拼:

  拼:  message.role, message.content, message.tool_calls
  不拼: finish_reason (你的代码判断用)
        refusal (你的代码判断用)
        annotations (UI 渲染用)
        usage (计费/限流用)
        id (日志追踪用)
        reasoning (仅当次生成可见, 不拼回)

════════════════════════════════════════════════════════════════════
Gemma 4 token 层面的 role 命名:

  API (你写的)   →  Token 层 (模型看到的)
  ─────────────      ─────────────────────
  system             system
  user               user
  assistant          model          ← Gemma 4 叫 model, 不是 assistant
  tool               tool

  chat_template 在 API ↔ token 之间做翻译, 你永远用标准 OpenAI role 名.

════════════════════════════════════════════════════════════════════
max_turns:
  - None = 无限循环, 靠模型返回文本自动停止
  - int  = 最多循环 N 轮, 超出强制结束 (防死循环)
  - 正常 agent 对话 2~3 轮就结束了, max_turns 只是安全网.
"""

from typing import cast

from openai import Omit, OpenAI, omit
from openai.types.chat import (
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessage,
  ChatCompletionSystemMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionMessageToolCallUnionParam,
  ChatCompletionToolUnionParam,
  ChatCompletionUserMessageParam,
)

from agent.config import VLLM_BASE_URL, MODEL_NAME
from agent.utils import show_token_mapping

_VERBOSE = True  # 是否打印每轮的工具调用详情


def _log(msg, border=False):
  if _VERBOSE:
    if border: print(f"{'─' * 50}")
    print(msg)
    if border: print(f"{'─' * 50}")



_CLIENT: OpenAI = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")  # 指向本地 vLLM
_MODEL: str = MODEL_NAME  # vLLM 模型名

def run_agent(
  sys_param: ChatCompletionSystemMessageParam,
  user_param: ChatCompletionUserMessageParam,
  tools: list[ChatCompletionToolUnionParam] | Omit = omit,  # 如果没 tools 就不传, tools=omit,
  *,
  max_turns: int | None = 5,  # 最多循环几轮; None=无限, 防止死循环
  max_tokens: int = 8192,
) -> str:
  """
  运行一次 agent loop, 返回模型的最终文本回复.

  Args:
    model:        模型名
    sys_param:    系统提示消息
    user_param:   用户消息
    max_turns:    最大 LLM 调用次数; None=无限 (生产慎用)
    max_tokens:   每次 LLM 调用的最大输出 token 数

  Returns:
    模型的最终文本回复 (str)

  数据流:
    ChatCompletionSystemMessageParam
    ChatCompletionUserMessageParam
    ChatCompletionAssistantMessageParam[..., tool_calls:ChatCompletionMessageToolCallUnionParam]
    ChatCompletionToolMessageParam
    ChatCompletionAssistantMessageParam
  """
  turn = 0
  msg_params = [
    sys_param,
    user_param,
  ]

  while max_turns is None or turn < max_turns:
    turn += 1

    _log(f"  Agent Turn {turn} / {max_turns}", border=True)

    # ── Step 1: 调用 LLM ──
    response: ChatCompletion = _CLIENT.chat.completions.create(
      model=_MODEL,
      messages=msg_params,
      tools=tools or omit,  # 如果没 tools 就不传, tools=omit,
      max_tokens=max_tokens,
      extra_body={
          "chat_template_kwargs": {"enable_thinking": True}
      },
   )

    msg: ChatCompletionMessage = response.choices[0].message

    # _log(response.to_json(indent=2))
    _log(f"[thinking]{msg.reasoning}")
    _log(f"[finish_reason] {(finish_reason := response.choices[0].finish_reason)}")
    _log(f"[usage] \n{response.usage.to_json(indent=2)}")

    # ── Step 2: 判断结果类型 ──
    #   A: tool_calls 不为空 → 执行工具函数, 追加 tool message, 回到循环开头 ->
    #   B: content 不为空 → 返回最终文本, loop 结束
    #   C: content 为空且 tool_calls 也为空 → 报错退出

    # 情况 A: 模型返回了工具调用
    if msg.tool_calls:
      _log(f"-> [tool calling] {len(msg.tool_calls)} tools")

      # Step A1: 把 tool_calls 转成 AssistantMessageParam, 追加到 msgs
      # (reasoning 仅打印, 不持久化; plan tool 负责显式记录推理步骤)
      msg_tc_params = cast(
        list[ChatCompletionMessageToolCallUnionParam],
        # list[ChatCompletionMessageCustomToolCallParam | ChatCompletionMessageFunctionToolCallParam]
        [tool_call.to_dict() for tool_call in msg.tool_calls],
     )
      asst_msg_param = ChatCompletionAssistantMessageParam(
        role=msg.role,
        content=msg.content,  # 通常是 None, for tool calling
        tool_calls=msg_tc_params,
     )
      msg_params += [asst_msg_param]

      # Step A2: 依次执行每个 tool, 把结果作为 tool message 追加
      from .tool_registry import execute_tool

      for tc in msg.tool_calls:
        if tc.type == "function":
          tool_name = tc.function.name
          tool_args = tc.function.arguments
          # 执行工具
          tool_result = execute_tool(tool_name, tool_args)
          _log(
            f"--> 执行{tc.to_dict()}工具: '{tool_name}', 参数: {tool_args}, id={tc.id}\n"
            f"--> 返回: {tool_result[:100]}..."
          )
            

          # 追加 tool message
          tool_msg_param = ChatCompletionToolMessageParam(
            content=tool_result,
            role="tool",
            tool_call_id=tc.id,
         )
          msg_params += [tool_msg_param]

        if tc.type == "custom":
          _log(f"  跳过非 function 工具调用: {tc.type}")
          continue

      # Step A3: 回到循环开头, 让 LLM 基于 tool 结果继续
      continue

    # 情况 B: 模型返回了普通文本 (可能 content 为 None 但 finish_reason="stop")
    if msg.content or finish_reason == "stop":
      msg_params += [ChatCompletionAssistantMessageParam(
        role=msg.role,
        content=msg.content,  # 可能是 None, 但 finish_reason="stop" 表示这是最终文本
      )]
      preview = (msg.content or "")[:100]
      _log(f"[final answer]\n{preview}...")
      show_token_mapping(msg_params, tools, add_generation_prompt=False)  # 显示 token mapping
      return msg.content or ""

    # 情况 C: content 为空且没有 tool_calls (不太常见, 但也要处理)
    _log(
      f"  -> [error] content is None and no tool_calls, finish_reason={finish_reason}"
   )
    return (
      msg.content
      or f"[error] finish_reason={finish_reason}, content is None and no tool_calls"
   )

  # 超出最大轮次
  _log(f"  -> [error] 超过最大轮次 {max_turns}, 未得到最终答案")
  if not msg_params:
    return ""

  last_msg = msg_params[-1]
  if isinstance(last_msg, dict):
    content = last_msg.get("content")
    return content if isinstance(content, str) else ""

  return ""
