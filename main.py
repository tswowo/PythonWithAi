import os

from dotenv import load_dotenv
from openai import OpenAI

import src.core
from src.tools import TOOLS

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
    print("程序开始")
    agent = src.core.Agent(client=client, tools=TOOLS)
    while True:
        current_message = input()
        if current_message == "exit" or current_message == "q":
            break
        try:
            if current_message.startswith("file "):
                parts = current_message.split(" ", 2)
                text = parts[2] if len(parts) > 2 else "总结这份文件"
                agent.chat_file(text=text, file_path=parts[1])
            else:
                agent.chat(current_message, stream=True, thinking=True)
            agent.save()
        except Exception as e:
            print(f"Error: {e}")
    print("退出程序")
