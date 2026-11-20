def make_content_tool(agent):
    def get_content_tokens():
        usage = getattr(agent, "usage", None)
        if usage is None:
            return "尚无用量读数，请先进行一轮对话"
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return get_content_tokens


CONTENT_TOKENS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_content_tokens",
        "description": "获取最近一次请求的token用量：prompt_tokens为发送的完整上下文大小，completion_tokens为该次回复大小，total_tokens为两者之和",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
