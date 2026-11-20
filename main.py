import os
import json

from dotenv import load_dotenv
from openai import OpenAI

import src.core
from src.tools import TOOLS, register_tool
from src.tools.rag import RagIndex, make_retrieve_tool

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

    rag = RagIndex()
    if os.path.exists("rag_index.json"):
        rag.load()
    register_tool("retrieve_knowledge", make_retrieve_tool(rag))

    agent = src.core.Agent(client=client, tools=TOOLS)
    if input("是否加载上次的对话？(load) ") == "load":
        with open("chat_history.json", "r", encoding="utf-8") as f:
            agent.messages = json.load(f)
    while True:
        current_message = input()
        if current_message == "exit" or current_message == "q":
            break
        try:
            if current_message.startswith("file "):
                parts = current_message.split(" ", 2)
                text = parts[2] if len(parts) > 2 else "总结这份文件"
                agent.chat_file(text=text, file_path=parts[1])
                agent.save()
            elif current_message.startswith("build "):
                folder = current_message.split(" ", 1)[1].strip()
                if not os.path.isdir(folder):
                    print(f"错误：文件夹不存在：{folder}")
                else:
                    rag.build(folder)
                    if rag.chunks:
                        rag.save()
                        print(f"知识库已更新，共 {len(rag.chunks)} 个 chunk，当前会话即可检索。")
            elif current_message.startswith("add "):
                folder = current_message.split(" ", 1)[1].strip()
                if not os.path.isdir(folder):
                    print(f"错误：文件夹不存在：{folder}")
                else:
                    before = len(rag.chunks)
                    rag.add(folder)
                    if len(rag.chunks) > before:
                        rag.save()
                        print(f"已追加 {len(rag.chunks) - before} 个 chunk，知识库共 {len(rag.chunks)} 个，当前会话即可检索。")
            else:
                agent.chat(current_message, stream=True, thinking=True)
                agent.save()
        except Exception as e:
            print(f"Error: {e}")
    print("退出程序")
