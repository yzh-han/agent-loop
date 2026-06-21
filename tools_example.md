# Tools example

```python
# 例子
# # ② JSON Schema 列表（发给 LLM 的 tools 参数）
# TOOLS: list[dict[str, Any]] = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "获取指定城市的当前天气信息",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "city": {
#                         "type": "string",
#                         "description": "城市名称，如 北京、上海、深圳",
#                     },
#                 },
#                 "required": ["city"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "calculate",
#             "description": "计算数学表达式，支持加减乘除和括号",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "expression": {
#                         "type": "string",
#                         "description": "数学表达式，如 '(1+2)*3'",
#                     },
#                 },
#                 "required": ["expression"],
#             },
#         },
#     },
# ]
```