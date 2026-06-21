"""
tools.py —— 具体 Tool 函数定义

在这里添加新工具：加一个函数 + @register_tool() 装饰器即可自动注册。
注册框架和执行入口在 tool_registry.py。

允许的参数注解规则见 tool_registry.py 文档。
"""

from typing import Annotated, Literal

from agent.tool_registry import register_tool


# ═══════════════════════════════════════════════════════════════════
# Tool 函数定义 —— 加一个 @register_tool() 就自动注册
# ═══════════════════════════════════════════════════════════════════


@register_tool()
def get_weather(city: Annotated[Literal["北京", "上海", "深圳", "杭州", "成都"], "城市名称"]) -> str:
    """获取指定城市的当前天气信息。"""
    weather_db = {
        "北京": "晴天, 25°C, 湿度40%, 空气质量优, 微风",
        "上海": "多云, 28°C, 湿度65%, 东南风3级",
        "深圳": "雷阵雨, 30°C, 湿度80%, 出门请带伞",
        "杭州": "阴天, 22°C, 湿度55%, 适合出游",
        "成都": "小雨, 18°C, 湿度70%, 体感微凉",
    }
    return weather_db.get(city, f"未找到「{city}」的天气数据，请换个城市试试。")


@register_tool(description="计算数学表达式，支持加减乘除和括号")
def calculate(expression: Annotated[str, "数学表达式，如 '(1+2)*3'"]) -> str:
    """计算一个数学表达式。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错: {e}"


@register_tool()
def plan(steps: Annotated[str, "行动计划, 每行一步, 如'1. 计算 187313+3213\n2. 判断奇偶\n3. 查天气\n4. 回答'"]) -> str:
    """制定行动计划。在调用其他工具之前, 先列出完整步骤清单。"""
    return f"计划已制定:\n{steps}"
