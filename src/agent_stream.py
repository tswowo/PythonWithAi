import src.tools


def exec_tool_call(messages, tool_call):
    output = src.tools.execute_tool(tool_call.function.name, tool_call.function.arguments)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": output,
    })


def do_chat_stream(client, messages, reasoning=False):
    if reasoning:
        do_chat_stream_reasoning(client, messages)
        return
    stream = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=True,
    )
    response_content = []
    for chunk in stream:
        chunk_message = chunk.choices[0].delta.content
        # 处理普通对话内容
        if chunk_message is not None:
            response_content.append(chunk_message)
            print(chunk_message, end="", flush=True)
    print()
    content = "".join(response_content)
    messages.append({"role": "assistant", "content": content})


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
