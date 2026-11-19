import os
from typing import Any

from src.logs import save_messages
import src.tools


class Agent:
    MODEL = "deepseek-flash"

    def __init__(self, client, model=MODEL, max_rounds=None, tools=None):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_rounds = max_rounds if max_rounds is not None else int(os.environ.get("MAX_TOOL_ROUNDS", "10"))
        self.messages: list[dict[str, Any]] = []

    # 输入层
    def chat(self, text=None, stream=False):
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        self._run_loop(stream)

    def chat_file(self, text, file_id, stream=False):
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                }, {
                    "type": "file",
                    "file_id": file_id
                }
            ]
        })
        self._run_loop(stream)

    # 循环层
    def _run_loop(self, stream):
        snapshot = len(self.messages)
        for _ in range(self.max_rounds):
            assistant_message = self._complete_stream() if stream else self._complete()
            self.messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                break
            self._run_tool_calls(tool_calls)
        else:
            del self.messages[snapshot:]
            print("工具调用轮次超限")

    def _run_tool_calls(self, tool_calls):
        for tc in tool_calls:
            output = src.tools.execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": output,
            })

    # 单轮层
    def _complete(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )
        message = response.choices[0].message
        if message.content:
            print(message.content)
        if message.tool_calls:
            print("调用工具中...")
        return message.model_dump(exclude_none=True)

    def _complete_stream(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            stream=True
        )
        content_parts = []
        tool_calls: dict[int, dict] = {}
        # 处理普通对话，收集工具调用信息
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 处理普通对话内容
            if delta.content is not None:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
            # 处理工具调用
            if delta.tool_calls:
                self._collect_tool_calls(tool_calls, delta.tool_calls)
        if content_parts:
            print()
        assistant_message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
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
