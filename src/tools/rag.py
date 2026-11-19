import json
import os
from typing import Any
import dashscope
from http import HTTPStatus

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_EMBEDDING_URL = "https://ws-a8tixj900b7cat2u.cn-beijing.maas.aliyuncs.com/api/v1"


class RagIndex:

    def __init__(self):
        self.chunks: list[dict[str, Any]] = []
        self.vectors = None

    @staticmethod
    def load_documents(folder_path):
        """从文件夹读取所有 txt 和 md 文件"""
        docs = []

        # 遍历文件夹下的所有文件
        for filename in os.listdir(folder_path):
            # 只处理文本文件
            if not filename.endswith(('.txt', '.md')):
                continue

            file_path = os.path.join(folder_path, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 只要文件里有内容，就加入列表
            if content.strip():
                docs.append({
                    "text": content,  # 文件全部内容
                    "source": filename  # 文件名作为来源
                })
        return docs

    @staticmethod
    def split_documents(docs, chunk_size=500, overlap=80):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            # 优先按段落、中文句号、问号等切分
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        print("开始分割文档")
        chunks = []
        for doc in docs:
            # 切割文本
            pieces = splitter.split_text(doc["text"])
            for piece in pieces:
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": piece,
                    "source": doc["source"]
                })
        print(f"共分割出 {len(chunks)} 个 chunk")
        return chunks

    @staticmethod
    def embedding_texts(texts):
        print("开始生成向量")
        vectors = []
        dashscope.base_http_api_url = BASE_EMBEDDING_URL
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            response = dashscope.TextEmbedding.call(
                model="qwen3.7-text-embedding",
                input=batch
            )
            if response.status_code != HTTPStatus.OK:
                raise Exception(f"向量化失败:{response.message}")
            items = sorted(response.output["embeddings"], key=lambda e: e["text_index"])
            vectors.extend(item["embedding"] for item in items)
        return vectors

    def build(self, folder_path):
        docs = self.load_documents(folder_path=folder_path)
        if not docs:
            print("没有找到任何文档")
            return
        self.chunks = self.split_documents(docs)
        texts = [chunk["text"] for chunk in self.chunks]
        self.vectors = self.embedding_texts(texts)
        for i, vector in enumerate(self.vectors):
            self.chunks[i]["vector"] = vector

    def save(self, path="rag_index.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=4)
        print(f"索引已保存到 {path}")

    def load(self, path="rag_index.json"):
        with open(path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"索引已加载从 {path}")
