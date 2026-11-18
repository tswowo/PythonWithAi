def do_chat(client, messages, reasoning=False):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    if reasoning:
        reasoning_content = response.choices[0].message.reasoning_content
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
            print(chunk_message, end="", flush=True)
    print()
    content = "".join(response_content)
    messages.append({"role": "assistant", "content": content})
