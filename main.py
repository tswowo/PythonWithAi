import os

from dotenv import load_dotenv
from openai import OpenAI
from src.agent import do_chat

if __name__ == '__main__':
    load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未读取到 API_KEY：请在 .env 中写入 API_KEY=你的密钥")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    messages = []
    print("程序开始")
    while (True):
        current_message = input()
        if (current_message == "exit" or current_message == "q"):
            break
        messages.append({"role": "user", "content": current_message})
        do_chat(client=client, messages=messages)
    print("退出程序")
