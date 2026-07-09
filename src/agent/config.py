"""
config.py —— 全局配置，从 .env 文件加载

用法：
    from agent.config import config
    print(config.MODEL_NAME)       # "gemma-4-E4B-it"
    print(config.VLLM_BASE_URL)    # "http://localhost:8000/v1"

为什么用 .env 而不是写死在代码里？
    1. 换模型只需改 .env，不用改代码
    2. 多人协作时每人有自己的 .env（不会被 git 追踪）
    3. API key 等敏感信息不会泄露到 git 历史
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 自动寻找项目根目录下的 .env 文件并加载到 os.environ
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


def _env(key: str, default: str = "") -> str:
    """读取环境变量，不存在时返回默认值。"""
    return os.getenv(key, default)


@dataclass
class Config:
    """全局配置，所有值从 .env 读取，有默认值。"""

    # ── vLLM 连接 ──
    VLLM_BASE_URL: str = field(
        default_factory=lambda: _env("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    VLLM_TOKENIZE_URL: str = field(
        default_factory=lambda: _env("VLLM_TOKENIZE_URL", "http://localhost:8000/tokenize")
    )
    VLLM_DETOKENIZE_URL: str = field(
        default_factory=lambda: _env("VLLM_DETOKENIZE_URL", "http://localhost:8000/detokenize")
    )
    
    API_KEY: str = field(
        default_factory=lambda: _env("API_KEY", "")
    )

    # ── 模型 ──
    MODEL_NAME: str = field(
        default_factory=lambda: _env("MODEL_NAME", "gemma-4-E4B-it")
    )

    # ── 采样参数 ──
    TEMPERATURE: float = field(
        default_factory=lambda: float(_env("TEMPERATURE", "0.7"))
    )
    MAX_TOKENS: int = field(
        default_factory=lambda: int(_env("MAX_TOKENS", "1024"))
    )

    # ── Agent ──
    AGENT_MAX_TURNS: int = field(
        default_factory=lambda: int(_env("AGENT_MAX_TURNS", "5"))
    )


# 全局单例 —— 整个项目 import 这一个就够了
config = Config()

VLLM_BASE_URL = config.VLLM_BASE_URL
VLLM_TOKENIZE_URL = config.VLLM_TOKENIZE_URL
VLLM_DETOKENIZE_URL = config.VLLM_DETOKENIZE_URL

MODEL_NAME = config.MODEL_NAME
API_KEY = config.API_KEY
