import json
import os
from http import HTTPStatus
from typing import Any

import dashscope
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_EMBEDDING_URL = "https://ws-a8tixj900b7cat2u.cn-beijing.maas.aliyuncs.com/api/v1"
EMBED_MODEL = "qwen3.7-text-embedding"
EMBED_BATCH_SIZE = 10


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
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            response = dashscope.TextEmbedding.call(
                model=EMBED_MODEL,
                input=batch
            )
            if response.status_code != HTTPStatus.OK:
                raise RuntimeError(f"向量化失败: {response.code} {response.message}")
            items = sorted(response.output["embeddings"], key=lambda e: e["text_index"])
            vectors.extend(item["embedding"] for item in items)
        return vectors

    def _refresh_matrix(self):
        """从 chunks 派生归一化矩阵，build/load 后统一调用"""
        matrix = np.array([c["vector"] for c in self.chunks], dtype=np.float32)
        self.vectors = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

    def build(self, folder_path):
        docs = self.load_documents(folder_path=folder_path)
        if not docs:
            print("没有找到任何文档")
            return
        self.chunks = self.split_documents(docs)
        texts = [chunk["text"] for chunk in self.chunks]
        vectors = self.embedding_texts(texts)
        for chunk, vector in zip(self.chunks, vectors):
            chunk["vector"] = vector
        self._refresh_matrix()

    def save(self, path="rag_index.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=4)
        print(f"索引已保存到 {path}")

    def load(self, path="rag_index.json"):
        with open(path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        self._refresh_matrix()
        print(f"索引已从 {path} 加载")

    def search(self, query, k=5, min_score=0.3):
        print(f"开始搜索: {query}")
        if self.vectors is None:
            return []
        query_vector = np.array(self.embedding_texts([query])[0], dtype=np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        scores = self.vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": round(score, 4)
            })
        print(f"搜索完成，找到 {len(results)} 条相关结果。")
        return results


RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_knowledge",
        "description": "检索本地知识库中的私有资料。当用户问题涉及内部文档、私有资料，或已有参考资料不足以回答时使用，建议用改写后的关键词检索。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或问题"
                },
                "k": {
                    "type": "integer",
                    "description": "返回条数，默认 5",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


def make_retrieve_tool(index):
    def retrieve_knowledge(query, k=5):
        if not index.chunks:
            return "知识库为空，请先建立索引"
        results = index.search(query, k=k)
        if not results:
            return "未找到相关资料"
        return "\n\n".join(
            f"[来源: {r['source']} | 相关度: {r['score']}]\n{r['text']}"
            for r in results
        )
    return retrieve_knowledge