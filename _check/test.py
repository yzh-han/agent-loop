
import json

from openai import Omit, OpenAI, omit
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessage, ChatCompletionMessageFunctionToolCallParam, ChatCompletionMessageParam, ChatCompletionToolUnionParam, ChatCompletionToolMessageParam

import httpx

from agent.tool_registry import TOOLS, execute_tool
import agent.tools
from agent.config import VLLM_BASE_URL, MODEL_NAME
from agent.utils import show_token_mapping

# =══════════════════════════════════════════════════════════════════
# 拦截 httpx 发出的请求
def log_request(request: httpx.Request):
    """在请求发出前，打印完整的 request body"""
    import json
    body = json.loads(request.content) if request.content else {}
    print("=== 实际发送的请求 ===")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Body 字段 ({len(body)} 个):")
    for k, v in sorted(body.items()):
        val = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else repr(v)
        print(f"  {k}: {val[:120]}")

http_client = httpx.Client(event_hooks={"request": [log_request]})
# =══════════════════════════════════════════════════════════════════
        
msgs = [] 
msgs += [
    {"role": "system", "content": "你是精炼的助手。"},
    {"role": "user", "content": "请问今天北京的天气如何?"},
]


# show_token_mapping(msgs, TOOLS)  # 显示 token mapping
 

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    # http_client=http_client,
)

response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        tools=TOOLS or omit,  # 如果没 tools 就不传, tools=omit,
        # tool_choice="none", # auto=模型自己选工具, none=不选工具, force=强制选工具
        # tool_choice={
        #     "type": "function",
        #     "function": {"name": "think"}
        # },
        extra_headers={
        "MY-ID": "sess-abc",         # 可覆盖 default_headers
        },
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        },
    )

msg: ChatCompletionMessage = response.choices[0].message

print("finish_reason:", response.choices[0].finish_reason)
print(msg.model_dump_json(indent=2))

if msg.tool_calls:
    print(f"\n=== 模型决定调用 {len(msg.tool_calls)} 个工具 ===")
    tool_calls = [
        ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.id,
            function=tool_call.function.to_dict(),
            type=tool_call.type,
        ) for tool_call in msg.tool_calls
    ]
    ass_msg_param = ChatCompletionAssistantMessageParam(
        role=msg.role,
        content=msg.content or msg.reasoning,
        tool_calls=tool_calls
    
    )
    msgs+= [ass_msg_param]
# print(msg.to_json(indent=2))
# print(json.dumps(ass_msg_param, indent=2))
# msgs+= [ass_msg_param]
print(json.dumps(msgs, indent=2))

show_token_mapping(msgs, TOOLS, add_generation_prompt=False)  # 显示 token mapping

tool_calls = msg.tool_calls
if tool_calls:
    print(f"\n=== 模型决定调用 {len(tool_calls)} 个工具 ===")
    for i, tool_call in enumerate(tool_calls):
        print(tool_call)
        function = tool_call.function
        print(f"  工具调用 {i+1}:")
        print(f"    name: {function.name}")
        print(f"    arguments: {function.arguments}")
        # 执行工具调用
        result = execute_tool(function.name, function.arguments)
        print(f"    执行结果: {result}, id={tool_call.id}")
        toolcall_output = ChatCompletionToolMessageParam(
            content=result,
            role="tool",
            tool_call_id=tool_call.id,
        )
        msgs+= [toolcall_output]
        show_token_mapping(msgs, TOOLS, add_generation_prompt=False)  # 显示 token mapping
        
response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        tools=TOOLS or omit,  # 如果没 tools 就不传, tools=omit,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        },
    )

print("\n=== 模型最终回答 ===")
msg: ChatCompletionMessage = response.choices[0].message
print(msg.model_dump_json(indent=2))