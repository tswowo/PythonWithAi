import json
from collections.abc import Callable

from .online_search import search_online, fetch_online, ONLINE_SEARCH_TOOL_SCHEMA, ONLINE_FETCH_TOOL_SCHEMA
from .rag import RETRIEVE_TOOL_SCHEMA
from .time import get_current_time, TIME_TOOL_SCHEMA

TOOLS = [
    TIME_TOOL_SCHEMA,
    ONLINE_SEARCH_TOOL_SCHEMA,
    ONLINE_FETCH_TOOL_SCHEMA,
    RETRIEVE_TOOL_SCHEMA
]

TOOL_MAP: dict[str, Callable[..., str]] = {
    "get_current_time": get_current_time,
    "search_online": search_online,
    "fetch_online": fetch_online,
}


def register_tool(name: str, func: Callable[..., str]) -> None:
    """运行时注册需要外部实例的工具（如依赖 RagIndex 的 retrieve_knowledge）。"""
    TOOL_MAP[name] = func


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
        return json.dumps(output, ensure_ascii=False)
    except Exception as e:
        print(f"错误：调用工具 '{name}' 时出错: {e}")
        return f"错误：调用工具 '{name}' 时出错: {e}"
