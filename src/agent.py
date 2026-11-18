import os

import src.tools

MODEL = "deepseek-flash"


def run_tool_calls(messages, tool_calls):
    for tool_call in tool_calls:
        output = src.tools.execute_tool(tool_call.function.name, tool_call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output,
        })


def create_completion(client, messages, tools):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    return response.choices[0].message


def do_chat(client, messages, reasoning=False):
    if reasoning:
        do_chat_reasoning(client, messages)
        return
    for _ in range(int(os.environ.get("MAX_TOOL_ROUNDS", 10))):
        message = create_completion(client, messages, src.tools.TOOLS)
        print(message.content if message.content else "调用工具中...")
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            break
        run_tool_calls(messages, message.tool_calls)
    else:
        messages.pop()
        print("工具调用轮次超限")


def do_chat_reasoning(client, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    reasoning_content = response.choices[0].message.reasoning_content
    print(reasoning_content)
    print(content)
