import json


def save_messages(messages, path="chat_history.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
