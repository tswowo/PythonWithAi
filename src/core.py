import os
from typing import Any

from src.files import build_file_message
from src.files import upload_file
from src.logs import save_messages
import src.tools

_USE_DEFAULT_TOOLS = object()


class Agent:
    MODEL = "deepseek-flash"

    def __init__(self, client, model=MODEL, max_rounds=None, tools=None):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_rounds = max_rounds if max_rounds is not None else int(os.environ.get("MAX_TOOL_ROUNDS", "10"))
        self.messages: list[dict[str, Any]] = []

    # 输入层
    def chat(self, text=None, stream=False, thinking=False):
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        self._run_loop(stream, thinking)

    def chat_file(self, text, file_path, stream=False, thinking=False):
        file_id = upload_file(self.client, file_path)
        self.messages.append(
            build_file_message(text, file_id)
        )
        self._run_loop(stream, thinking)

    # 循环层
    def _run_loop(self, stream, thinking):
        for _ in range(self.max_rounds):
            assistant_message = self._complete_stream(thinking, _USE_DEFAULT_TOOLS) if stream else self._complete(
                thinking)
            self.messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                break
            self._run_tool_calls(tool_calls)
        else:
            print("工具调用轮次超限")
            self.messages.append({
                "role": "user",
                "content": "工具调用轮次暂时已达上限。",
            })
            assistant_message = self._complete_stream(thinking=thinking) if stream else self._complete(
                thinking=thinking)
            self.messages.append(assistant_message)

    def _run_tool_calls(self, tool_calls):
        for tc in tool_calls:
            output = src.tools.execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": output,
            })

    # 单轮层
    def _complete(self, thinking, tools=_USE_DEFAULT_TOOLS):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools if tools is _USE_DEFAULT_TOOLS else tools,
            extra_body={
                "thinking": {
                    "type": "enabled" if thinking else "disabled"
                }}
        )
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            print(reasoning)
        if message.content:
            print(message.content)
        if message.tool_calls:
            print("调用工具中...")
        return message.model_dump(exclude_none=True)

    def _complete_stream(self, thinking, tools=_USE_DEFAULT_TOOLS):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools if tools is _USE_DEFAULT_TOOLS else tools,
            stream=True,
            extra_body={
                "thinking": {
                    "type": "enabled" if thinking else "disabled"
                }}
        )
        content_parts = []
        reasoning_parts = []
        tool_calls: dict[int, dict] = {}
        # 处理普通对话，收集工具调用信息
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 处理思维链
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is not None:
                reasoning_parts.append(reasoning)
                print(reasoning, end="", flush=True)
            # 处理普通对话内容
            if delta.content is not None:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
            # 处理工具调用
            if delta.tool_calls:
                self._collect_tool_calls(tool_calls, delta.tool_calls)
        if reasoning_parts or content_parts:
            print()
        assistant_message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            assistant_message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls.values()
            ]
        return assistant_message

    @staticmethod
    def _collect_tool_calls(tool_calls, delta_tool_calls):
        for tc in delta_tool_calls:
            if tc.index not in tool_calls:
                tool_calls[tc.index] = {"id": "", "name": "", "arguments": ""}
            entry = tool_calls[tc.index]
            if tc.id:
                entry["id"] = tc.id
            if tc.function and tc.function.name:
                entry["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                entry["arguments"] += tc.function.arguments

    # 持久化
    def save(self, path="chat_history.json"):
        save_messages(self.messages, path)
