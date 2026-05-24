"""
query_test.py
=============
向 Pinecone 查詢最相關法規段落，用於驗證 index 正確，也是 RAG 的 retrieval 核心。

使用方式：
    python query_test.py                  # 互動模式
    python query_test.py "食品添加物的規定"  # 直接查詢

輸出：Top-K 最相關段落 + 來源檔案路徑
"""

import sys
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone

# ─────────────────────────────────────────────
# ★ 與 build_index.py 相同的設定
# ─────────────────────────────────────────────
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"   # ← 填入 Pinecone API Key
PINECONE_INDEX   = "food-rag"

TOP_K = 5   # 回傳前 K 筆結果


# ═══════════════════════════════════════════════
# 載入模型（單例，避免重複載入）
# ═══════════════════════════════════════════════

_model = None

def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        print("載入 BAAI/bge-m3...")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("模型載入完成 ✅\n")
    return _model


# ═══════════════════════════════════════════════
# 查詢函式
# ═══════════════════════════════════════════════

def query(question: str, top_k: int = TOP_K) -> list:
    """
    輸入問題，回傳 top_k 個最相關段落。

    Returns:
        [{"score": float, "source": str, "chunk_id": int, "text": str}, ...]
    """
    model = get_model()

    # Embed 問題
    result = model.encode(
        [question],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    query_vec = result["dense_vecs"][0].tolist()

    # 向 Pinecone 查詢
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response["matches"]:
        meta = match.get("metadata", {})
        results.append({
            "score":    round(match["score"], 4),
            "source":   meta.get("source", ""),
            "chunk_id": meta.get("chunk_id", -1),
            "text":     meta.get("text", ""),
        })
    return results


def print_results(question: str, results: list):
    print(f"\n{'═'*60}")
    print(f"問題：{question}")
    print(f"{'═'*60}")
    for i, r in enumerate(results, 1):
        print(f"\n【第 {i} 名】相似度 {r['score']}  |  來源：{r['source']}")
        print(f"{'-'*50}")
        # 顯示前 300 字
        preview = r["text"][:300].replace("\n", " ")
        print(preview + ("..." if len(r["text"]) > 300 else ""))
    print(f"\n{'═'*60}\n")


# ═══════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        # 命令列直接傳入問題
        question = " ".join(sys.argv[1:])
        results  = query(question)
        print_results(question, results)
    else:
        # 互動模式
        print("🔍 RAG 法規查詢測試（輸入 'q' 離開）\n")
        get_model()  # 預先載入
        while True:
            question = input("請輸入問題：").strip()
            if question.lower() in ("q", "quit", "exit", ""):
                print("離開。")
                break
            results = query(question)
            print_results(question, results)


if __name__ == "__main__":
    main()
