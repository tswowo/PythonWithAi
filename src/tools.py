import datetime

import json
from collections.abc import Callable


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

TOOL_MAP: dict[str, Callable[..., str]] = {
    "get_current_time": get_current_time
}


def execute_tool(name, arguments):
    print(f"调用函数{name}，参数{arguments}")
    func = TOOL_MAP.get(name)
    if func is None:
        print(f"错误：调用了不存在的工具 '{name}'")
        return f"错误：调用了不存在的工具 '{name}'"
    try:
        args = json.loads(arguments) if arguments else {}
        output = func(**args)
        print(f"调用函数{name}，参数{arguments}，返回结果{output}")
        return output
    except Exception as e:
        print(f"错误：调用工具 '{name}' 时出错: {e}")
        return f"错误：调用工具 '{name}' 时出错: {e}"
