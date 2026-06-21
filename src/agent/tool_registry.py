"""
tool_registry.py —— Tool 注册框架

提供：
    - register_tool 装饰器：自动把函数注册到 TOOL_FUNCTIONS 和 TOOLS
    - execute_tool：agent loop 调用工具的入口
    - TOOLS：发给 LLM 的 JSON Schema 列表
    - TOOL_FUNCTIONS：函数名 → 可调用对象的注册表

允许的参数注解规则（用于自动生成 JSON Schema）：
    - 基础类型：str / int / float / bool
    - Annotated[T, "描述"]：在 T 的 schema 上追加 description
    - Literal[...]：统一转成 string enum
    - Optional[T] 或 T | None：仅允许这种 Union 形式
    - list[T]：递归生成 items
    - dict[str, T]：递归生成 additionalProperties（key 必须是 str）

不符合规则的注解会在注册阶段直接报错。
"""

import json
import inspect
from openai.types.chat import ChatCompletionToolParam, ChatCompletionToolUnionParam
from types import UnionType
from typing import Any, Annotated, Literal, Union, get_args, get_origin, get_type_hints

# ═══════════════════════════════════════════════════════════════════
# Python type → JSON Schema type 映射
# ═══════════════════════════════════════════════════════════════════

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation) -> dict[str, Any]:
    """
    把 Python type annotation 转成 JSON Schema 类型定义。

    允许规则（不符合会直接抛 ValueError):
        - 基础类型:str / int / float / bool
        - Annotated[T, "描述"]：在 T 的 schema 上追加 description
        - Literal[...]：统一转成 string enum
        - Optional[T] 或 T | None:仅允许这种 Union 形式
        - list[T]：递归生成 items
        - dict[str, T]：递归生成 additionalProperties(key 必须是 str)
    """
    origin = get_origin(annotation)

    # 处理 Annotated[type, description]
    if origin is Annotated:
        base_type, *metadata = get_args(annotation)
        schema = _python_type_to_json_schema(base_type)
        for m in metadata:
            if isinstance(m, str):
                schema["description"] = m
                break
        return schema

    # 处理 Literal["a", "b"] → enum（统一转成 string）
    if origin is Literal:
        values = [str(v) for v in get_args(annotation)]
        if not values:
            raise ValueError("Literal[...] 不能为空。")
        return {"type": "string", "enum": values}

    # 处理 Union / Optional（PEP 604: X | Y）
    if origin in (Union, UnionType):
        args = get_args(annotation)
        non_none_args = [a for a in args if a is not type(None)]

        # Optional[X] = Union[X, None]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            return _python_type_to_json_schema(non_none_args[0])

        # 对 agent tool 参数禁用一般 Union，避免 LLM 参数分支歧义
        raise ValueError(
            f"不支持 Union 类型注解: {annotation}. "
            "仅允许 Optional[T] 或 T | None。"
        )

    # 处理 list[T]
    if origin is list:
        args = get_args(annotation)
        if len(args) != 1:
            raise ValueError(f"list 注解必须是 list[T] 形式，当前为: {annotation}")
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    # 处理 dict[str, T]
    if origin is dict:
        args = get_args(annotation)
        if len(args) != 2:
            raise ValueError(f"dict 注解必须是 dict[str, T] 形式，当前为: {annotation}")
        key_type, value_type = args
        if key_type is not str:
            raise ValueError(
                f"dict 的 key 类型必须是 str，当前为: {key_type}（注解: {annotation}）"
            )
        return {"type": "object", "additionalProperties": _python_type_to_json_schema(value_type)}

    # 基础类型
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    raise ValueError(
        f"不支持的类型注解: {annotation}. "
        "仅允许 str/int/float/bool、Annotated、Literal、Optional、list[T]、dict[str, T]。"
    )


def _make_json_schema(func) -> ChatCompletionToolParam:
    """
    从函数签名自动生成 OpenAI tool 的 JSON Schema。

    Returns:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    name = func.__name__
    doc = (func.__doc__ or "").strip()
    description = doc.split("\n")[0].strip() if doc else name

    hints = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    def _is_optional_annotation(annotation: Any) -> bool:
        origin = get_origin(annotation)
        if origin is Annotated:
            base_type, *_ = get_args(annotation)
            return _is_optional_annotation(base_type)
        if origin in (Union, UnionType):
            return any(arg is type(None) for arg in get_args(annotation))
        return False

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        if param_name not in hints:
            raise ValueError(
                f"函数 '{name}' 的参数 '{param_name}' 缺少类型注解。"
            )

        annotation = hints[param_name]
        is_optional = (
            _is_optional_annotation(annotation)
            or param.default is not inspect.Parameter.empty
        )

        properties[param_name] = _python_type_to_json_schema(annotation)
        if not is_optional:
            required.append(param_name)

    return ChatCompletionToolParam({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    })


# ═══════════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════════

_TOOL_FUNCTIONS: dict[str, Any] = {} # 函数名 → 可调用对象
TOOLS: list[ChatCompletionToolUnionParam] = []    # JSON Schema 列表，发给 LLM


def register_tool(description: str = ""):
    """
    装饰器：自动注册一个 tool 函数。

    自动做的事：
        1. 把 func 加入 _TOOL_FUNCTIONS(按函数名索引）
        2. 从函数签名 + docstring 生成 JSON Schema, 追加到 TOOLS
        3. 如果传了 description 参数，覆盖 docstring 的描述

    用法：
        @register_tool()
        def get_weather(city: str) -> str:
            '''获取指定城市的天气。'''
            ...

        @register_tool(description="计算数学表达式")
        def calculate(expression: str) -> str:
            ...
    """

    def decorator(func):
        schema = _make_json_schema(func)
        if description:
            schema["function"]["description"] = description

        _TOOL_FUNCTIONS[func.__name__] = func
        TOOLS.append(schema)
        return func

    return decorator


# ═══════════════════════════════════════════════════════════════════
# Tool 执行入口（agent loop 调用）
# ═══════════════════════════════════════════════════════════════════

def execute_tool(name: str, arguments: str) -> str:
    """
    Agent loop 调用此函数执行工具。

    Args:
        name:      工具名（来自 LLM 的 tool_calls[0].function.name）
        arguments: 参数 JSON string（来自 LLM 的 tool_calls[0].function.arguments）

    Returns:
        工具的字符串返回值
    """
    if name not in _TOOL_FUNCTIONS:
        return f"错误：未知工具 '{name}'"

    try:
        args_dict = json.loads(arguments)
    except json.JSONDecodeError:
        return f"错误：无法解析工具参数 JSON: {arguments}"

    if not isinstance(args_dict, dict):
        return f"错误：工具参数必须是 JSON object, 实际收到: {type(args_dict).__name__}"

    try:
        result = _TOOL_FUNCTIONS[name](**args_dict)
        return str(result)
    except TypeError as e:
        return f"错误：参数不匹配: {e}"
    except Exception as e:
        return f"错误：工具执行异常: {e}"


__all__ = ["register_tool", "execute_tool", "TOOLS"]