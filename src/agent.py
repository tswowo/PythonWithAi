def do_chat(client, messages):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    content = response.choices[0].message.content
    response_finish_reason = response.choices[0].finish_reason
    print(content)
    messages.append({"role": "assistant", "content": content, "finish_reason": response_finish_reason})


def do_chat_stream(client, messages):
    stream = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=True,
    )
    response_content = []
    response_finish_reason = ""
    for chunk in stream:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is None:
            continue
        response_content.append(chunk_message)
        response_finish_reason = chunk.choices[0].finish_reason
        print(chunk_message, end="", flush=True)
    print()
    content = "".join(response_content)
    messages.append({"role": "assistant", "content": content, "finish_reason": response_finish_reason})
