"""
Embed supplemental extraction outputs with Hugging Face and upsert to Pinecone.

This avoids the local FlagEmbedding/Torch stack when local package versions are
temporarily incompatible, while keeping the same BAAI/bge-m3 embedding model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pinecone import Pinecone


EXTRACTED_DIR = Path("extracted_texts")
SUPPLEMENTAL_REPORT = Path("supplemental_processing_report.csv")
CHECKPOINT_FILE = Path("embed_checkpoint.json")
PINECONE_INDEX = "food-rag"
CHUNK_MAX_CHARS = 600
CHUNK_MIN_CHARS = 30
UPSERT_BATCH = 100


def chunk_text(text: str, source: str) -> list[dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    article_pattern = r"(?=第\s*[一二三四五六七八九十百千0-9]+\s*條)"
    segments = re.split(article_pattern, text)
    if len(segments) <= 1:
        segments = re.split(r"\n\s*\n", text)

    raw_chunks: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > CHUNK_MAX_CHARS:
            sub_segs = [s.strip() for s in seg.split("\n") if s.strip()]
            buf = ""
            for sub in sub_segs:
                if len(buf) + len(sub) > CHUNK_MAX_CHARS and len(buf) >= CHUNK_MIN_CHARS:
                    raw_chunks.append(buf)
                    buf = sub
                else:
                    buf = f"{buf}\n{sub}" if buf else sub
            if buf:
                raw_chunks.append(buf)
        else:
            raw_chunks.append(seg)

    chunks = [
        c for c in raw_chunks
        if len(c) >= CHUNK_MIN_CHARS and re.search(r"[\u4e00-\u9fff\w]", c)
    ]
    return [{"text": c, "chunk_id": i, "source": source} for i, c in enumerate(chunks)]


def make_vector_id(source: str, chunk_id: int) -> str:
    h = hashlib.md5(source.encode()).hexdigest()[:12]
    return f"{h}_{chunk_id}"


def l2_normalize(vec: Iterable[float]) -> list[float]:
    arr = np.asarray(list(vec), dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 1e-9 else arr.tolist()


def embed_text(client: InferenceClient, text: str) -> list[float]:
    result = client.feature_extraction(text, model="BAAI/bge-m3")
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr[0]
    elif arr.ndim == 3:
        vec = arr[0][0]
    else:
        raise RuntimeError(f"Unsupported embedding shape: {arr.shape}")
    if vec.shape[0] != 1024:
        raise RuntimeError(f"Expected 1024 dimensions, got {vec.shape[0]}")
    return l2_normalize(vec.tolist())


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")))
    return set()


def save_checkpoint(done: set[str]) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def supplemental_outputs() -> list[Path]:
    with SUPPLEMENTAL_REPORT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [
        Path(row["輸出檔"])
        for row in rows
        if row["輸出檔"] and row["新狀態"].startswith("✅")
    ]
    return sorted(set(paths))


def main() -> None:
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)
    client = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])
    done = load_checkpoint()

    pending = [path for path in supplemental_outputs() if str(path) not in done]
    print(f"Supplemental outputs: {len(supplemental_outputs())}")
    print(f"Pending upsert: {len(pending)}")

    buffer: list[dict] = []
    upserted = 0

    def flush() -> None:
        nonlocal upserted
        if not buffer:
            return
        index.upsert(vectors=buffer)
        upserted += len(buffer)
        buffer.clear()

    for path in pending:
        text = path.read_text(encoding="utf-8", errors="replace")
        source = str(path.relative_to(EXTRACTED_DIR))
        chunks = chunk_text(text, source)
        print(f"{source}: {len(chunks)} chunks")
        for chunk in chunks:
            for attempt in range(3):
                try:
                    vec = embed_text(client, chunk["text"])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            buffer.append({
                "id": make_vector_id(chunk["source"], chunk["chunk_id"]),
                "values": vec,
                "metadata": {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:1500],
                },
            })
            if len(buffer) >= UPSERT_BATCH:
                flush()
        flush()
        done.add(str(path))
        save_checkpoint(done)

    stats = index.describe_index_stats()
    print(f"Upserted vectors: {upserted}")
    print(f"Index total vectors: {stats.get('total_vector_count')}")


if __name__ == "__main__":
    main()
