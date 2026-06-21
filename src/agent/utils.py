
import httpx
import json
    
from agent.config import config

MODEL_NAME = config.MODEL_NAME
VLLM_BASE_URL = config.VLLM_BASE_URL
VLLM_TOKENIZE_URL = config.VLLM_TOKENIZE_URL
VLLM_DETOKENIZE_URL = config.VLLM_DETOKENIZE_URL

def show_token_mapping(
    messages: list[dict],
    tools = None,
    tool_choice: str = "auto", # 默认是 auto，表示模型自己选工具；none=不选工具；force=强制选工具
    add_generation_prompt: bool = True,
):
    client = httpx.Client(timeout=30.0)

    # ---- 第 1 步：调用 /tokenize endpoint，看看 token IDs ----
    # 注意：vLLM 的 tokenize endpoint 在 /tokenize（不在 /v1 下）
    resp = client.post(
        VLLM_TOKENIZE_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,  # vLLM 会先用 chat_template 拼接，再 tokenize
            "tools": tools,  # 如果有 tools，就传给 vLLM
            "add_generation_prompt": add_generation_prompt,  # 加上 <start_of_turn>assistant\n
            "return_token_strs": True,
            "tool_choice": tool_choice,  # auto=模型自己选工具, none=不选工具, force=强制选工具
        },
    )
    tokenize_result: dict = resp.json()
    """
    tokenize_result 的结构：
    {
        count: 123,
        max_model_len: 65536,
        tokens: [101, 102, 103, ...],  # int list
        token_strs: ["<start_of_turn>", "assistant", "\n", ...],  # str list
    }
    print(json.dumps(tokenize_result, indent=2, ensure_ascii=False))  # 打印原始 JSON       
    """
    token_ids = tokenize_result["tokens"]  # 字段名是 "tokens"，是 int list

    # ---- 第 2 步：调用 /detokenize endpoint，看看这些 token 对应的文本 ----
    # 注意：vLLM 的 detokenize endpoint 在 /detokenize（不在 /v1 下）
    resp = client.post(
        VLLM_DETOKENIZE_URL,
        json={"model": MODEL_NAME, "tokens": token_ids},
    )
    """
    resp.json() 的结构：
    {
        "prompt": "<start_of_turn>assistant\n你是一个实用的助手..."
    }
    """
    prompt_text = resp.json()["prompt"]

    print("-- mapping " + "-"* 49)
    # 把特殊 token 标记出来，方便观察 chat template 的结构
    # 替换不可见的特殊 token 为可见标记
    display_text = (
        prompt_text
        .replace("\n", "↵\n")
    )
    
    display_text = (
        display_text
        .replace('<tool|>', '\n<tool|>')
    )
    display_text = (
        display_text
        .replace('<|tool>', '\n<|tool>')
    )
    display_text = (
        display_text
        .replace('<tool_call|>', '<tool_call|>\n')
    )
    display_text = (
        display_text
        .replace('<tool_response|>', '<tool_response|>\n')
    )
    # display_text = (
    #     display_text
    #     .replace('<|"|>', '\n<|"|>')
    # )
    print(display_text)
    print("-" * 60)
    
if __name__ == "__main__":
    # 测试用的 messages
    test_messages = [
        {"role": "system", "content": "你是一个实用的助手。"},
        {"role": "user", "content": "请问今天的天气如何？"},
    ]
    show_token_mapping(test_messages)