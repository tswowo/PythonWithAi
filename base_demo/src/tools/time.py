import datetime


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TIME_TOOL_SCHEMA ={
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取当前时间",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
