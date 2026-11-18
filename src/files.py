TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".py", ".xlsx", ".xls", ".epub", ".mobi"}
DOC_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}


def upload_file(client, path, rich=True):
    # 解析文件扩展名
    ext = "." + path.rsplit(".", 1)[-1].lower()
    if ext in DOC_EXTS:
        purpose = "rich_layout_recognition" if rich else "text_recognition"
    elif ext in TEXT_EXTS:
        purpose = "user_data"
    else:
        raise ValueError(f"不支持的文件扩展名：{ext}")
    # 上传文件到服务端
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose=purpose).id


def build_file_message(text, file_id):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text
            }, {
                "type": "file",
                "file_id": file_id
            }
        ]
    }
