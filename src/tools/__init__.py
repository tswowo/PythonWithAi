import json
from collections.abc import Callable

from .online_search import search_online, fetch_online
from .time import get_current_time

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
    },
    {
        "type": "function",
        "function": {
            "name": "search_online",
            "description": "按 query 联网搜索网页，返回结构化结果。由 DeepSeek 自研搜索引擎提供支持，适用于需要实时信息、新闻或事实核查的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，1～128 个字符，不能全为空白，支持 site:、lang:、after: 等搜索语法",
                        "minLength": 1,
                        "maxLength": 128
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回结果条数，1～50，默认 10",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    },
                    "ori_question": {
                        "type": "string",
                        "description": "触发搜索的用户原始问题，1～128 个字符，有助于理解搜索意图，不确定时可不传",
                        "maxLength": 128
                    },
                    "localization": {
                        "type": "string",
                        "description": "用于优先返回本地化搜索结果的 IP 地址，默认为空字符串",
                        "default": ""
                    },
                    "need_content": {
                        "type": "boolean",
                        "description": "是否在 content 中返回网页正文，默认 false；需要阅读正文时传 true",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_online",
            "description": "抓取单个网页，以 Markdown 文本返回页面内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页地址，长度不超过 8192，必须为含 host 的 HTTP/HTTPS URL，且不能携带用户名或密码。"
                    },
                },
                "required": ["url"]
            }
        }
    },
]

TOOL_MAP: dict[str, Callable[..., str]] = {
    "get_current_time": get_current_time,
    "search_online": search_online,
    "fetch_online": fetch_online,
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
        return str(output)
    except Exception as e:
        print(f"错误：调用工具 '{name}' 时出错: {e}")
        return f"错误：调用工具 '{name}' 时出错: {e}"
