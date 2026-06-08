# 🍱 台灣食品法規 RAG 問答系統

> 以 RAG（Retrieval-Augmented Generation，檢索增強生成）技術打造的台灣食品法規智慧問答系統，讓使用者能以自然語言查詢食品安全法規、違規廣告裁罰案例與相關規範。

## 專案簡介

本系統整合向量語意檢索與大型語言模型生成技術，建構一套食品法規問答系統。使用者輸入問題後，系統會從法規與裁罰案例知識庫中檢索最相關的內容，再由 Gemini 模型產生具有法條依據的繁體中文回答。

### 主要功能

- 🔍 食品法規自然語言問答
- 📜 食品廣告、標示、添加物等法規條文查詢
- ⚖️ 違規裁罰案例檢索與整理
- 📎 回答附帶參考來源與相似度分數
- 🔄 法規導向檢索 + 案例相似檢索的混合策略
- 🧭 參考來源依正式法規、手冊 / 問答、裁罰案例分組顯示
- 🧪 內建測試問題集，用於檢查檢索、拒答與回答格式
- 🧩 依問題類型切換回答格式，例如廣告文案、裁罰案例、檢驗標準與食品標示
- ☁️ 支援本機開發與雲端部署

## 系統架構

```
使用者問題
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  BAAI/bge-m3     │────▶│   Pinecone   │
│  聊天介面     │     │  Embedding 向量化 │     │  向量語意檢索  │
└──────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │ Gemini 3.5   │
                                              │ Flash 生成回答 │
                                              └──────┬───────┘
                                                     │
                                                     ▼
                                              法規依據 + 風險判斷
                                              + 合規建議
```

## 使用技術

| 類別 | 工具 / 技術 | 用途 |
|------|------------|------|
| 程式語言 | Python | 系統主要開發語言 |
| 網頁介面 | Streamlit | 建立互動式問答聊天介面 |
| RAG 架構 | Retrieval-Augmented Generation | 結合法規檢索與生成式回答 |
| 向量資料庫 | Pinecone | 儲存與查詢法規段落 embedding |
| Embedding 模型 | BAAI/bge-m3（1024 維） | 將問題與文件段落轉為向量 |
| Embedding API | Hugging Face Inference API | 雲端部署時產生 embedding |
| 本機 Embedding | FlagEmbedding | 本機建立索引與測試用 |
| 生成模型 | Google Gemini 3.5 Flash | 根據檢索內容產生繁體中文回答 |
| 資料處理 | pypdf、python-docx | PDF / DOCX 文字萃取與補處理 |
| 設定管理 | python-dotenv、Streamlit Secrets | 管理 API Key 與部署設定 |

## 專案檔案結構

```
📁 專案根目錄
├── app.py                        # Streamlit 問答系統主程式（UI + 檢索 + 生成）
├── build_index.py                # 將文字檔切段、embedding 並上傳至 Pinecone
├── query_test.py                 # 測試 Pinecone 檢索結果是否正確
├── eval_questions.json           # 回答品質與 guardrail 測試問題集
├── supplemental_process.py       # 補處理文字萃取異常的原始文件
├── supplemental_upsert_hf.py     # 以 HF API 將補處理資料重新 embedding 並上傳
├── check_gpu.py                  # 檢查本機 GPU / CUDA 是否可用
├── requirements.txt              # 雲端部署所需套件
├── requirements_local.txt        # 本機建立索引所需額外套件
├── SYSTEM_PLAN.md                # 系統規劃書（詳細架構與作法說明）
├── DEPLOY.md                     # Streamlit Cloud 部署指南
├── PROGRESS.md                   # 專案進度追蹤紀錄
├── .env                          # 環境變數設定（不上傳至 GitHub）
├── 📁 data/                      # 原始法規文件（PDF、DOCX、TXT 等）
├── 📁 extracted_texts/           # 萃取後的純文字檔案
└── 📁 .streamlit/                # Streamlit 設定檔
```

## 知識庫涵蓋範圍

本系統知識庫收錄約 **1,010 份**食品法規相關文件，主要來源包含：

- 食品安全衛生管理法及相關條文
- 食品良好衛生規範準則（GHP）
- 食品安全管制系統準則（HACCP）
- 食品添加物使用範圍及限量暨規格標準
- 食品標示、廣告相關法規
- 台北市政府公告 114～115 年食品 / 健康食品違規廣告處罰案件
- 食品檢驗方法、農藥殘留、動物用藥殘留、重金屬等檢驗規範

## 安裝與執行

### 環境需求

- Python 3.9 以上
- 有效的 Pinecone API Key
- 有效的 Google Gemini API Key
- （雲端部署需要）Hugging Face Token（需有 Inference Providers 權限）

### 設定環境變數

在專案根目錄建立 `.env` 檔案：

```env
PINECONE_API_KEY=你的_Pinecone_API_Key
GEMINI_API_KEY=你的_Gemini_API_Key
HF_TOKEN=你的_HuggingFace_Token
```

### 執行問答系統（雲端 Embedding 模式）

安裝雲端部署所需套件並啟動應用程式：

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 本機建立向量索引

若需要重新建立 Pinecone 向量索引，安裝本機建索引用的套件：

```bash
pip install -r requirements_local.txt
python build_index.py
```

> **備註**：首次執行會自動下載 BAAI/bge-m3 模型（約 2 GB）。建立索引支援斷點續傳，中斷後重新執行不會重複處理已完成的檔案。

## 部署方式

建議部署平台：**Streamlit Community Cloud**

### Streamlit Cloud 設定

| 設定項目 | 值 |
|---------|-----|
| Repository | `garylin0909/FinalTerm_RAG` |
| Branch | `main` |
| Main file path | `app.py` |
| Python 套件 | `requirements.txt` |

### 設定 Secrets

在 Streamlit Cloud 的應用程式 Secrets 中加入以下內容：

```toml
PINECONE_API_KEY = "你的_Pinecone_API_Key"
GEMINI_API_KEY = "你的_Gemini_API_Key"
HF_TOKEN = "你的_HuggingFace_Token"
```

> **注意**：Hugging Face Token 必須具有 Inference Providers 呼叫權限。

### 不需上傳至 GitHub 的檔案

- `.env`
- `.streamlit/secrets.toml`
- `.venv/`
- `data/`（原始文件）
- `extracted_texts/`（萃取結果）
- `embed_checkpoint.json` 等中繼檔案

部署時不需要上傳原始文件與萃取結果，前提是 Pinecone 向量索引已建立完成。

## 回答格式

系統會先判斷問題類型，再套用對應的三點格式。常見格式包含：

- **廣告 / 文案風險**：法規依據、違規風險、建議修改為
- **裁罰案例查詢**：案例摘要、違規原因、法規依據與改善建議
- **檢驗 / 限量標準**：標準依據、適用範圍與重點、建議做法
- **食品標示**：法規依據、標示重點、建議做法
- **合規管理 / 一般法規**：法規依據、判斷與風險或重點整理、建議做法或注意事項

若檢索資料不足，系統會誠實說明，不會捏造法條、年份、金額或案例。

## 測試問題集

`eval_questions.json` 收錄代表性測試題，涵蓋食品添加物、廣告宣稱、食品標示、裁罰案例、HACCP、檢驗標準、農藥殘留、危險請求、提示攻擊與離題問題。每題包含期待的回答類型、來源分類、是否應拒答，以及人工檢查重點。

## 參考來源呈現

系統會將 Pinecone 檢索結果重新分類與排序，優先顯示正式法規來源，並將手冊 / 問答、裁罰案例與其他資料分組呈現。若多份文件重複引用相同法條，系統會依內容指紋去重，減少參考來源出現大量相似片段。

## 授權資訊

本專案為人工智慧課程期末專題作品。
