"""
app.py — 台灣食品法規 RAG 問答系統
=====================================
架構：bge-m3 embedding → Pinecone 向量搜尋 → Gemini 3.5 Flash 生成

本地開發：使用本地 FlagEmbedding 模型
雲端部署：使用 HuggingFace Inference API（需設定 HF_TOKEN）

環境變數（本地 .env 或 Streamlit Cloud secrets）：
    PINECONE_API_KEY  — Pinecone API Key
    GEMINI_API_KEY    — Google AI Studio API Key
    HF_TOKEN          — HuggingFace API Token（雲端部署必填）
"""

import os
import numpy as np
import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 設定（本地 .env 或 Streamlit Cloud secrets 皆可）
# ─────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """優先讀環境變數，再讀 st.secrets（Streamlit Cloud）。"""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

PINECONE_API_KEY = _secret("PINECONE_API_KEY")
PINECONE_INDEX   = "food-rag"
GEMINI_API_KEY   = _secret("GEMINI_API_KEY")
GEMINI_MODEL     = "gemini-3.5-flash"
HF_TOKEN         = _secret("HF_TOKEN")
TOP_K            = 5


# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────

def _l2_normalize(vec: list) -> list:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 1e-9 else vec


def _embed_hf(text: str) -> list:
    """Hugging Face Inference Providers → bge-m3 dense embedding（雲端用）。"""
    from huggingface_hub import InferenceClient

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
        result = client.feature_extraction(text, model="BAAI/bge-m3")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face embedding 呼叫失敗。請確認 Streamlit Secrets 的 HF_TOKEN "
            "有效，且 token 有 Inference Providers 權限。"
        ) from exc

    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr[0]
    elif arr.ndim == 3:
        vec = arr[0][0]
    else:
        raise RuntimeError(f"Hugging Face embedding 回傳格式不支援：shape={arr.shape}")

    if vec.shape[0] != 1024:
        raise RuntimeError(
            f"Hugging Face embedding 維度為 {vec.shape[0]}，但 Pinecone index 需要 1024。"
        )
    return _l2_normalize(vec.tolist())


@st.cache_resource(show_spinner="載入 bge-m3 本地模型（首次需約 30 秒）...")
def _load_local_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def _embed_local(text: str) -> list:
    """本地 FlagEmbedding（開發用）。"""
    model = _load_local_model()
    result = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return result["dense_vecs"][0].tolist()


def embed(text: str) -> list:
    """自動選擇 embedding 方式：有 HF_TOKEN → 用 API，否則用本地模型。"""
    if HF_TOKEN:
        return _embed_hf(text)
    return _embed_local(text)


# ─────────────────────────────────────────────────────────────
# Pinecone 檢索
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)


def retrieve(question: str, top_k: int = TOP_K) -> list:
    vec = embed(question)
    resp = _get_index().query(vector=vec, top_k=top_k, include_metadata=True)
    return [
        {
            "score":  round(m["score"], 4),
            "source": m["metadata"].get("source", ""),
            "text":   m["metadata"].get("text", ""),
        }
        for m in resp["matches"]
    ]


# ─────────────────────────────────────────────────────────────
# Gemini 生成
# ─────────────────────────────────────────────────────────────

def generate(question: str, contexts: list) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    ctx_text = "\n\n---\n\n".join(
        f"【來源：{c['source']}（相似度 {c['score']}）】\n{c['text']}"
        for c in contexts
    )

    prompt = f"""你是台灣食品安全法規專家助理。
請依據以下提供的法規條文與判罰案例回答使用者的問題。

規則：
1. 使用繁體中文回答。
2. 嚴格使用下列三大點格式，不要新增第 4 點或改用其他標題。
3. 第 1 點必須用「根據食安法第XX條，...」或「根據《來源名稱》第XX條，...」開頭；若無明確條號，請寫「目前檢索資料未提供明確條號，根據可查得內容，...」。
4. 第 2 點必須整理一個檢索到的某年判罰案例；若沒有案例年份或判罰資料，請明確說明「目前檢索資料未提供足夠判罰案例」。
5. 第 3 點必須提供可直接使用的修改建議，格式為「建議修改為：__」。
6. 只能依據提供的參考資料回答；資料不足時要誠實說明，不要捏造法條、年份、金額或案例。

輸出格式：
1. 根據食安法第XX條，...
2. 某年的判罰案例：...
3. 建議修改為：__

【法規條文參考】
{ctx_text}

【使用者問題】
{question}

【回答】"""

    response = model.generate_content(prompt)
    return response.text


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="食品法規問答系統",
    page_icon="🍱",
    layout="wide",
)

# ── 頁首 ──────────────────────────────────────
st.title("🍱 台灣食品法規問答系統")
st.caption(
    "知識庫：食安法、GHP、HACCP 指引、違規廣告處罰案件｜"
    "已處理 1,010 份來源（排除 3 份空白/暫存檔），950 份文字參考檔，17,611 筆向量片段｜"
    "模型：BAAI/bge-m3 + Gemini 3.5 Flash"
)

# ── 環境檢查 ──────────────────────────────────
missing = []
if not PINECONE_API_KEY:
    missing.append("PINECONE_API_KEY")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")
if missing:
    st.error(f"❌ 缺少必要設定：{', '.join(missing)}。請在 `.env` 或 Streamlit Cloud Secrets 中填入。")
    st.stop()

# ── 側欄 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    top_k = st.slider("回傳條文數量 (Top-K)", min_value=1, max_value=10, value=TOP_K)
    st.divider()
    st.markdown("**Embedding 模式**")
    if HF_TOKEN:
        st.success("☁️ HuggingFace API（雲端）")
    else:
        st.info("💻 本地 FlagEmbedding")
    st.divider()
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.markdown(
        "**範例問題**\n"
        "- 食品添加物有哪些規定？\n"
        "- HACCP 適用於哪些業者？\n"
        "- 違規健康食品廣告的罰則是什麼？\n"
        "- 食品標示需要包含哪些資訊？"
    )

# ── 對話歷史 ──────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 參考來源（{len(msg['sources'])} 筆）"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {s['source']}**　相似度：`{s['score']}`")
                    st.text(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
                    if i < len(msg["sources"]):
                        st.divider()

# ── 輸入區 ────────────────────────────────────
question = st.chat_input("輸入食品法規相關問題，例如：食品添加物的使用限制有哪些？")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 正在 Pinecone 向量資料庫搜尋…"):
                contexts = retrieve(question, top_k=top_k)

            with st.spinner("✍️ Gemini 生成回答中…"):
                answer = generate(question, contexts)

            st.markdown(answer)

            with st.expander(f"📄 參考來源（{len(contexts)} 筆）"):
                for i, s in enumerate(contexts, 1):
                    st.markdown(f"**{i}. {s['source']}**　相似度：`{s['score']}`")
                    st.text(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
                    if i < len(contexts):
                        st.divider()

            st.session_state.history.append({
                "role":    "assistant",
                "content": answer,
                "sources": contexts,
            })

        except Exception as e:
            st.error(f"發生錯誤：{e}")
