def do_chat(client, messages):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    messages.append(response.choices[0].message)
    print(response.choices[0].message.content)