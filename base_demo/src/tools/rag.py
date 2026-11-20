import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import dashscope
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_EMBEDDING_URL = "https://ws-a8tixj900b7cat2u.cn-beijing.maas.aliyuncs.com/api/v1"
EMBED_MODEL = "qwen3.7-text-embedding"
EMBED_BATCH_SIZE = 20 # 百炼平台要求的每批向量最大数量
EMBED_WORKERS = 5  # 并发线程数
EMBED_MAX_RETRIES = 8  # 单批限流重试次数，退避等待 5s→10s→…→60s 封顶
EMBED_CACHE_BIN = "embed_cache.bin"  # 断点续传：已完成批次的向量
EMBED_CACHE_META = "embed_cache.json"  # 断点续传：输入指纹与已完成批数


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
        print(f"共读取 {len(docs)} 个文件")
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
    def _embed_batch(batch):
        """单批向量化，供线程池并发调用；触发限流时自动退避重试"""
        for attempt in range(EMBED_MAX_RETRIES):
            response = dashscope.TextEmbedding.call(
                model=EMBED_MODEL,
                input=batch
            )
            if response.status_code == HTTPStatus.OK:
                items = sorted(response.output["embeddings"], key=lambda e: e["text_index"])
                return [item["embedding"] for item in items]
            if "Throttling" in str(response.code) and attempt < EMBED_MAX_RETRIES - 1:
                wait = min(5 * 2 ** attempt, 60)
                print(f"触发限流，{wait}s 后重试（{attempt + 1}/{EMBED_MAX_RETRIES - 1}）")
                time.sleep(wait)
                continue
            raise RuntimeError(f"向量化失败: {response.code} {response.message}")

    @staticmethod
    def embedding_texts(texts):
        print("开始生成向量")
        dashscope.base_http_api_url = BASE_EMBEDDING_URL
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
        total = len(batches)
        # 单批场景（如 search 查询）不走断点缓存，避免误覆盖建库缓存
        if total <= 1:
            vectors = []
            for batch in batches:
                vectors.extend(RagIndex._embed_batch(batch))
            return vectors

        # 输入指纹：文本完全一致才复用上次已生成的向量
        fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        vectors = []
        done = 0
        if os.path.exists(EMBED_CACHE_META) and os.path.exists(EMBED_CACHE_BIN):
            with open(EMBED_CACHE_META, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fingerprint:
                done = meta.get("done_batches", 0)
                cached = np.fromfile(EMBED_CACHE_BIN, dtype=np.float32)
                vectors = cached.reshape(-1, meta["dim"]).tolist()
                print(f"从断点续传：已完成 {done}/{total} 批")
        with open(EMBED_CACHE_BIN, "ab" if done else "wb") as cache_f:
            # map 按提交顺序返回结果，保证向量与 texts 一一对应
            with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
                for i, batch_vectors in enumerate(pool.map(RagIndex._embed_batch, batches[done:]), done + 1):
                    vectors.extend(batch_vectors)
                    # 每批完成即落盘，中断后下次可从断点继续
                    np.asarray(batch_vectors, dtype=np.float32).tofile(cache_f)
                    cache_f.flush()
                    with open(EMBED_CACHE_META, "w", encoding="utf-8") as f:
                        json.dump({"fingerprint": fingerprint, "done_batches": i, "dim": len(batch_vectors[0])}, f)
                    if i % 50 == 0 or i == total:
                        print(f"向量化进度: {i}/{total} 批")
        # 全部完成，清理断点缓存
        os.remove(EMBED_CACHE_BIN)
        os.remove(EMBED_CACHE_META)
        return vectors

    def _refresh_matrix(self):
        """从 chunks 派生归一化矩阵，build/load 后统一调用"""
        if not self.chunks:
            self.vectors = None
            return
        matrix = np.array([c["vector"] for c in self.chunks], dtype=np.float32)
        self.vectors = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

    def _embed_and_split(self, docs):
        """切分并向量化一批文档，返回带向量的新 chunk"""
        new_chunks = self.split_documents(docs)
        texts = [chunk["text"] for chunk in new_chunks]
        vectors = self.embedding_texts(texts)
        for chunk, vector in zip(new_chunks, vectors):
            chunk["vector"] = vector
        return new_chunks

    def build(self, folder_path):
        """全量重建：用文件夹内容替换整个知识库"""
        docs = self.load_documents(folder_path=folder_path)
        if not docs:
            print("没有找到任何文档")
            return
        new_chunks = self._embed_and_split(docs)
        # 向量化成功后才替换内存，中途失败时原有索引保持完整
        self.chunks = new_chunks
        self._refresh_matrix()

    def add(self, folder_path):
        """增量追加：把文件夹内容加入现有知识库"""
        docs = self.load_documents(folder_path=folder_path)
        if not docs:
            print("没有找到任何文档")
            return
        new_chunks = self._embed_and_split(docs)
        # 向量化成功后才追加，中途失败时原有索引保持完整
        base = len(self.chunks)
        for j, chunk in enumerate(new_chunks):
            chunk["chunk_id"] = base + j
        self.chunks.extend(new_chunks)
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