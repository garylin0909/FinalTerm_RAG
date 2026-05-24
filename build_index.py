"""
build_index.py
==============
將 extracted_texts/ 下所有 .txt 檔案切段、embedding，上傳至 Pinecone。

使用方式：
    python build_index.py

依賴安裝：
    pip install -r requirements.txt

首次執行會自動下載 BAAI/bge-m3 模型（約 2 GB）。
斷點續傳：透過 embed_checkpoint.json 記錄已完成的檔案，重跑不會重複 upsert。
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict

from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv() 
# ─────────────────────────────────────────────
# ★ 請填入你的設定
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")   # ← 填入 Pinecone API Key
PINECONE_INDEX   = "food-rag"                # Index 名稱（不存在會自動建立）
PINECONE_CLOUD   = "aws"                     # Serverless cloud（免費方案用 aws）
PINECONE_REGION  = "us-east-1"              # Serverless region

EXTRACTED_DIR    = Path("extracted_texts")   # 相對於本腳本的路徑
CHECKPOINT_FILE  = Path("embed_checkpoint.json")

# ─────────────────────────────────────────────
# 切段參數
# ─────────────────────────────────────────────
CHUNK_MAX_CHARS = 600   # 段落最大字元數（超過再切）
CHUNK_MIN_CHARS = 30    # 太短的段落過濾掉

# ─────────────────────────────────────────────
# 模型 / 批次參數
# ─────────────────────────────────────────────
EMBED_BATCH   = 64    # embed 每批張數（GPU 建議 64，CPU 建議 8~16）
UPSERT_BATCH  = 100   # Pinecone upsert 每批筆數
DIMENSION     = 1024  # bge-m3 固定輸出維度（勿更改）


# ═══════════════════════════════════════════════
# 切段邏輯
# ═══════════════════════════════════════════════

# 條文編號模式：第一條、第 2 條、第十二項、第三款 …
ARTICLE_RE = re.compile(
    r'(?=第\s*[一二三四五六七八九十百千零\d]+\s*[條項款目章節附])'
)


def chunk_text(text: str, source: str) -> List[Dict]:
    """
    依條文編號切割；若無條文結構則依雙換行切割。
    若某段落超過 CHUNK_MAX_CHARS，再進一步依單換行切成子段落。

    回傳：
        [{"text": str, "chunk_id": int, "source": str}, ...]
    """
    # 先嘗試條文切割
    segments = ARTICLE_RE.split(text)

    # 沒有條文結構 → 段落切割
    if len(segments) <= 1:
        segments = re.split(r'\n{2,}', text)

    raw_chunks: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 超長段落再切（依單換行）
        if len(seg) > CHUNK_MAX_CHARS:
            sub_segs = [s.strip() for s in seg.split('\n') if s.strip()]
            buf = ""
            for sub in sub_segs:
                if len(buf) + len(sub) > CHUNK_MAX_CHARS and len(buf) >= CHUNK_MIN_CHARS:
                    raw_chunks.append(buf)
                    buf = sub
                else:
                    buf = (buf + "\n" + sub).strip() if buf else sub
            if buf:
                raw_chunks.append(buf)
        else:
            raw_chunks.append(seg)

    # 過濾雜訊（太短、全為符號/空白）
    chunks = [
        c for c in raw_chunks
        if len(c) >= CHUNK_MIN_CHARS and re.search(r'[一-鿿\w]', c)
    ]

    return [
        {"text": c, "chunk_id": i, "source": source}
        for i, c in enumerate(chunks)
    ]


def make_vector_id(source: str, chunk_id: int) -> str:
    """以 source 路徑 hash + chunk_id 生成唯一 ID（避免特殊字元問題）"""
    h = hashlib.md5(source.encode()).hexdigest()[:12]
    return f"{h}_{chunk_id}"


# ═══════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ═══════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════

def main():
    # ── 1. 連線 Pinecone，確保 index 存在 ────────────────────
    print("連線 Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_names = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing_names:
        print(f"  建立新 index: {PINECONE_INDEX} (dim={DIMENSION}, cosine)")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # 等待 ready
        for _ in range(30):
            if pc.describe_index(PINECONE_INDEX).status["ready"]:
                break
            time.sleep(2)
        print("  Index 建立完成 ✅")
    else:
        print(f"  Index '{PINECONE_INDEX}' 已存在，直接使用")

    index = pc.Index(PINECONE_INDEX)

    # ── 2. 載入 bge-m3 ───────────────────────────────────────
    print("\n載入 BAAI/bge-m3（首次需下載約 2 GB）...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("  模型載入完成 ✅")

    # ── 3. 掃描所有 .txt ─────────────────────────────────────
    txt_files = sorted(EXTRACTED_DIR.rglob("*.txt"))
    print(f"\n找到 {len(txt_files)} 份文字檔")

    done_files = load_checkpoint()
    remaining  = [f for f in txt_files if str(f) not in done_files]
    print(f"已完成: {len(done_files)} | 待處理: {len(remaining)}\n")

    if not remaining:
        print("所有檔案已完成，無需重新 embed。")
        print(f"Index 統計: {index.describe_index_stats()}")
        return

    # ── 4. 切段 → Embed → Upsert ─────────────────────────────
    total_vectors = 0
    upsert_buffer: List[Dict] = []
    failed_files: List[str] = []

    def flush_buffer():
        nonlocal total_vectors
        if upsert_buffer:
            index.upsert(vectors=upsert_buffer)
            total_vectors += len(upsert_buffer)
            upsert_buffer.clear()

    for fpath in tqdm(remaining, desc="建立向量索引", unit="檔"):
        try:
            text = fpath.read_text(encoding="utf-8").strip()
            if not text:
                done_files.add(str(fpath))
                continue

            rel_source = str(fpath.relative_to(EXTRACTED_DIR))
            chunks = chunk_text(text, rel_source)

            if not chunks:
                done_files.add(str(fpath))
                continue

            # Embed（分批）
            texts = [c["text"] for c in chunks]
            for i in range(0, len(texts), EMBED_BATCH):
                batch_texts  = texts[i : i + EMBED_BATCH]
                batch_chunks = chunks[i : i + EMBED_BATCH]

                result = model.encode(
                    batch_texts,
                    batch_size=EMBED_BATCH,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense_vecs = result["dense_vecs"]

                for chunk, vec in zip(batch_chunks, dense_vecs):
                    upsert_buffer.append({
                        "id":     make_vector_id(rel_source, chunk["chunk_id"]),
                        "values": vec.tolist(),
                        "metadata": {
                            "source":   chunk["source"],
                            "chunk_id": chunk["chunk_id"],
                            # Pinecone metadata 值上限 ~40 KB；截斷保護
                            "text":     chunk["text"][:1500],
                        },
                    })

                if len(upsert_buffer) >= UPSERT_BATCH:
                    flush_buffer()

            done_files.add(str(fpath))
            save_checkpoint(done_files)

        except Exception as e:
            tqdm.write(f"⚠️  失敗: {fpath.name} — {e}")
            failed_files.append(str(fpath))
            continue

    # 剩餘 buffer
    flush_buffer()

    # ── 5. 完成報告 ───────────────────────────────────────────
    print(f"\n✅ 完成！共 upsert {total_vectors} 筆向量")
    stats = index.describe_index_stats()
    print(f"   Index 向量總數: {stats['total_vector_count']}")

    if failed_files:
        print(f"\n⚠️  失敗 {len(failed_files)} 份（可重跑，checkpoint 會跳過已完成的）：")
        for f in failed_files:
            print(f"   {f}")

    # 更新 PROGRESS.md
    _update_progress()


def _update_progress():
    """在 PROGRESS.md 更新向量資料庫建立狀態"""
    prog = Path("PROGRESS.md")
    if not prog.exists():
        return
    content = prog.read_text(encoding="utf-8")
    content = content.replace(
        "- [ ] 建立向量資料庫（embedding）",
        f"- [x] 建立向量資料庫（embedding）— 完成，模型: BAAI/bge-m3，向量庫: Pinecone({PINECONE_INDEX})"
    )
    prog.write_text(content, encoding="utf-8")
    print("PROGRESS.md 已更新 ✅")


if __name__ == "__main__":
    main()
