import os
import requests

DS_Search_URL = "https://api.deepseek.com/v1/tools/search"
DS_FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"


def search_online(query: str, count: int = 10, ori_question: str = "", localization: str = "",
                  need_content: bool = False):
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    response = requests.post(
        DS_Search_URL,
        headers={
            "Authorization": f"Bearer {deepseek_api_key}",
        },
        json={
            "query": query,
            "count": count,
            "ori_question": ori_question,
            "localization": localization,
            "need_content": need_content,
        },
        timeout=30
    )
    response.raise_for_status()

    result = response.json()
    print(result)
    return result


def fetch_online(url: str):
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    response = requests.post(
        DS_FETCH_URL,
        headers={
            "Authorization": f"Bearer {deepseek_api_key}",
        },
        json={"url": url},
        timeout=30
    )
    response.raise_for_status()

    page = response.json()
    print(page)
    return page


ONLINE_SEARCH_TOOL_SCHEMA = {
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
}
ONLINE_FETCH_TOOL_SCHEMA = {
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
}
