import os
from typing import Any

import src.tools

MODEL = "deepseek-flash"


def run_tool_calls(messages, tool_calls):
    for tool_call in tool_calls:
        output = src.tools.execute_tool(tool_call["function"]["name"], tool_call["function"]["arguments"])
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": output,
        })


def collect_tool_calls(tool_calls, delta_tool_calls):
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


def create_completion(client, messages, tools):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        tools=tools
    )


def do_chat_stream(client, messages, reasoning=False):
    if reasoning:
        do_chat_stream_reasoning(client, messages)
        return
    for _ in range(int(os.environ.get("MAX_TOOL_ROUNDS", 10))):
        stream = create_completion(client, messages, src.tools.TOOLS)
        response_content = []
        tool_calls = {}
        # 处理普通对话，收集工具调用信息
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 处理普通对话内容
            if delta.content is not None:
                response_content.append(delta.content)
                print(delta.content, end="", flush=True)
            # 处理工具调用
            if delta.tool_calls:
                collect_tool_calls(tool_calls, delta.tool_calls)
        if response_content:
            print()
        assistant_message: dict[str, Any] = {"role": "assistant", "content": "".join(response_content)}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls.values()
            ]
        messages.append(assistant_message)
        if not tool_calls:
            break
        print("调用工具中...")
        run_tool_calls(messages, assistant_message["tool_calls"])
    else:
        messages.pop()
        print("工具调用轮次超限")


def do_chat_stream_reasoning(client, messages):
    stream = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=True,
    )
    response_content = []
    response_reasoning_content = []
    for chunk in stream:
        chunk_message = chunk.choices[0].delta.content
        # 处理思维链内容
        if chunk.choices[0].delta.reasoning_content is not None:
            reasoning_content = chunk.choices[0].delta.reasoning_content
            response_reasoning_content.append(reasoning_content)
            print(reasoning_content, end="", flush=True)
        # 处理普通对话内容
        if chunk_message is not None:
            response_content.append(chunk_message)
            print(chunk_message, end="", flush=True)
    print()
    content = "".join(response_content)
    messages.append({"role": "assistant", "content": content})
