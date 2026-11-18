def do_chat(client, messages, reasoning=False):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    content = response.choices[0].message.content
    response_finish_reason = response.choices[0].finish_reason
    messages.append({"role": "assistant", "content": content, "finish_reason": response_finish_reason})
    if reasoning:
        reasoning_content = response.choices[0].message.reasoning_content
        messages[-1]["reasoning_content"] = reasoning_content
        print(reasoning_content)
    print(content)


def do_chat_stream(client, messages, reasoning=False):
    stream = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=True,
    )
    response_content = []
    response_reasoning_content = []
    response_finish_reason = ""
    for chunk in stream:
        chunk_message = chunk.choices[0].delta.content
        # 处理思维链内容
        if reasoning and chunk.choices[0].delta.reasoning_content is not None:
            reasoning_content = chunk.choices[0].delta.reasoning_content
            response_reasoning_content.append(reasoning_content)
            print(reasoning_content, end="", flush=True)
        # 处理普通对话内容
        if chunk_message is not None:
            response_content.append(chunk_message)
            response_finish_reason = chunk.choices[0].finish_reason
            print(chunk_message, end="", flush=True)
    print()
    content = "".join(response_content)
    messages.append({"role": "assistant", "content": content, "finish_reason": response_finish_reason})
    if reasoning:
        messages[-1]["reasoning_content"] = "".join(response_reasoning_content)
