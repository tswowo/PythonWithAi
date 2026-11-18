import os

from dotenv import load_dotenv
from openai import OpenAI

from agent_stream import do_chat_stream
from files import upload_file, build_file_message
from src.logs import save_messages

BASE_URL_DEEPSEEK = "https://api.deepseek.com"

if __name__ == '__main__':
    load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未读取到 API_KEY：请在 .env 中写入 API_KEY=你的密钥")
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL_DEEPSEEK
    )
    messages = []
    print("程序开始")
    while True:
        current_message = input()
        if current_message == "exit" or current_message == "q":
            break
        messages.append({"role": "user", "content": current_message})
        try:
            if current_message.startswith("file "):
                parts = current_message.split(" ", 2)
                file_id = upload_file(client, parts[1])
                text = parts[2] if len(parts) > 2 else "总结这份文件"
                messages.append(build_file_message(text, file_id))
            else:
                messages.append({"role": "user", "content": current_message})

            do_chat_stream(client=client, messages=messages)
        except Exception as e:
            print(f"Error: {e}")
        save_messages(messages)

    print("退出程序")
