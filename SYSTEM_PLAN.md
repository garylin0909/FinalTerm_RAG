# 台灣食品法規 RAG 問答系統規劃書

## 1. 系統目標

本專案建置一套「台灣食品法規問答系統」，使用 RAG（Retrieval-Augmented Generation，檢索增強生成）架構，讓使用者能以自然語言詢問食品安全、食品廣告、GHP、HACCP、檢驗方法與違規裁罰案例等問題。系統會先從法規與案例知識庫中檢索相關內容，再交由大型語言模型產生有依據的繁體中文回答。

## 2. 系統架構與詳細作法

本系統採用 RAG 架構，分成「資料建置流程」與「線上問答流程」兩部分。資料建置流程負責把食品法規文件整理成可檢索的向量資料庫；線上問答流程負責接收使用者問題、檢索相關內容，並由生成式模型回答。

### 2.1 原始資料蒐集

作法：

1. 將食藥署與台北市政府公告等資料放入 `data/` 資料夾。
2. 資料類型包含 PDF、DOCX、DOC、TXT 等格式。
3. 資料內容涵蓋食品安全衛生管理法、GHP、HACCP、檢驗方法、食品標示、食品廣告與違規裁罰案例。
4. 以資料夾分類保存來源，例如食藥署資料、台北市政府 114 年與 115 年違規廣告資料。

目的：

將不同來源與不同格式的食品法規資料集中管理，作為後續建立知識庫的基礎。

### 2.2 文字萃取與資料清理

作法：

1. 將原始文件轉成純文字並輸出到 `extracted_texts/`。
2. 對一般 PDF 或文字型文件，使用文字抽取工具取得內容。
3. 對圖片型 PDF，使用 OCR 方式補萃取文字。
4. 對萃取結果過短、表格格式不完整或錯誤的檔案，使用 `supplemental_process.py` 進行補處理。
5. PDF 補處理會使用 `pypdf` 重新抽取一般文字與 layout-preserving 文字。
6. DOCX 補處理會使用 `python-docx` 抽取段落與表格；若是圖片型 DOCX，則嘗試使用同資料夾中已成功萃取的對應檔案內容。
7. 補處理結果會更新到 `extraction_report.csv` 與 `supplemental_processing_report.csv`。

目的：

確保進入向量資料庫的內容是可讀、可檢索且具有法規語意的文字，降低空白文件、亂碼、表格遺失或 OCR 失敗造成的回答錯誤。

### 2.3 文件切段

作法：

1. 使用 `build_index.py` 中的 `chunk_text()` 函式處理 `extracted_texts/` 下的文字檔。
2. 優先依照法規常見格式切段，例如「第 1 條」、「第十二條」、「第 3 項」等條文結構。
3. 若文件沒有明顯條文格式，則依照空行或段落分隔切段。
4. 若單一段落超過 `CHUNK_MAX_CHARS = 600`，再依單行文字進一步切成較小段落。
5. 若段落小於 `CHUNK_MIN_CHARS = 30`，或內容幾乎都是符號，則過濾掉避免雜訊進入資料庫。
6. 每個 chunk 會保留 `source`、`chunk_id` 與文字內容，方便回答時顯示來源。

目的：

將長篇法規文件切成適合 embedding 與檢索的段落。切段太長會降低檢索精準度，切段太短則容易失去語意，因此本系統用條文結構與字數限制取得平衡。

### 2.4 Embedding 向量化

作法：

1. 使用 `BAAI/bge-m3` 作為 embedding 模型。
2. 本地建立索引時，透過 `FlagEmbedding` 載入 `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)`。
3. 每批最多處理 `EMBED_BATCH = 64` 筆文字，可搭配 GPU 加速。
4. 每個 chunk 會轉換成 1024 維 dense vector。
5. 雲端或補處理 upsert 時，使用 Hugging Face Inference API 產生同一模型的 embedding。
6. 透過 `embed_checkpoint.json` 記錄已完成的檔案，避免中斷後重跑時重複處理。

目的：

將文字語意轉換成向量，使系統可以用語意相似度搜尋，而不是只能依靠關鍵字比對。使用同一個 embedding 模型處理文件與問題，可以讓檢索結果更一致。

### 2.5 向量資料庫建立

作法：

1. 使用 Pinecone 作為向量資料庫。
2. Index 名稱為 `food-rag`。
3. 向量維度設定為 `DIMENSION = 1024`，對應 `BAAI/bge-m3` 的輸出維度。
4. 相似度計算方式使用 cosine similarity。
5. 若 Pinecone index 不存在，`build_index.py` 會自動建立 serverless index。
6. 每筆向量包含：
   - `id`：由來源路徑 hash 加上 `chunk_id` 組成，避免中文或特殊字元造成 ID 問題。
   - `values`：embedding 向量。
   - `metadata.source`：來源檔案路徑。
   - `metadata.chunk_id`：段落編號。
   - `metadata.text`：段落文字，最多保留約 1500 字。
7. 每累積 `UPSERT_BATCH = 100` 筆向量就批次上傳到 Pinecone。

目的：

建立可快速語意搜尋的食品法規知識庫，讓線上問答時不需要重新掃描原始文件，只要查詢 Pinecone 即可取得相關段落。

### 2.6 使用者問題處理

作法：

1. 使用者在 Streamlit 聊天輸入框輸入問題。
2. `app.py` 會先檢查 `PINECONE_API_KEY` 與 `GEMINI_API_KEY` 是否存在。
3. 系統將使用者問題送入 `embed()` 函式產生問題向量。
4. 若環境中有 `HF_TOKEN`，使用 Hugging Face Inference API 產生 embedding，適合 Streamlit Cloud 部署。
5. 若沒有 `HF_TOKEN`，則使用本地 `FlagEmbedding` 模型產生 embedding，適合本機開發。

目的：

將使用者自然語言問題轉換成與知識庫相同格式的向量，讓系統可以進行語意檢索。

### 2.7 法規與案例檢索

作法：

1. 系統不是只查詢使用者問題本身，而是分成兩種檢索：
   - 法規導向檢索：依使用者問題與關鍵主題動態產生法規查詢，例如食品廣告、食品標示、食品添加物、HACCP、GHP 或衛生管理等。
   - 案例相似檢索：使用使用者原始問題查詢相似裁罰案例或相關文件。
2. 法規導向檢索使用 `LAW_TOP_K = 4`。
3. Pinecone 第一階段會抓取較多候選資料：法規導向候選使用 `LAW_CANDIDATE_K = 8`，案例相似候選使用 `CASE_CANDIDATE_K = 12`。
4. 兩組結果合併後，會先依來源與文字內容指紋去除重複片段，降低不同手冊重複引用同一法條造成的雜訊。
5. 系統會將來源分類為正式法規、手冊 / 問答、裁罰案例與其他資料。
6. 第二階段 reranking 會綜合原始向量相似度、來源類型、問題類型、關鍵詞重疊、片語命中與主題加權，重新排序候選片段。
7. 食品添加物相關問題會優先檢索與排序「食品添加物使用範圍及限量暨規格標準」等正式規範。
8. 最多保留 `MAX_CONTEXTS = 12` 筆參考資料。
9. 每筆參考資料會標註來源分類、檢索類型、來源、相似度、重排分數與文字內容。

目的：

避免系統只找到案例卻缺少法規依據，或只找到手冊引用而沒有正式法規原文。透過「法規導向 + 案例相似」的混合檢索方式，可以讓回答同時具備正式依據、實務案例與可讀性。

### 2.8 Prompt 組合與回答生成

作法：

1. `app.py` 將檢索到的段落整理成 `ctx_text`。
2. 每段參考資料都會包含來源分類、檢索類型、來源與相似度。
3. 系統會使用 `classify_question_type()` 判斷問題類型，例如廣告 / 文案風險、裁罰案例查詢、檢驗 / 限量標準、食品標示、合規管理或一般法規查詢。
4. `answer_format_for()` 會依問題類型選擇三點回答格式。
5. 系統 prompt 會要求 Gemini 扮演「台灣食品安全法規專家助理」。
6. Prompt 明確限制回答規則：
   - 必須使用繁體中文。
   - 必須依選定的三點格式回答，不可新增第 4 點。
   - 不同問題類型使用不同標題，例如「建議修改為」、「案例摘要」、「標準依據」、「標示重點」或「注意事項」。
   - 不可捏造法條、年份、金額、限量、檢驗方法或案例。
   - 資料不足時要明確說明。
   - 手冊、問答或指引只能作為輔助說明，不應取代正式法規。
7. 生成模型使用 `gemini-3.5-flash`。
8. 最後由 `model.generate_content(prompt)` 產生回答文字。

目的：

透過明確的 prompt 約束回答格式與資料來源，降低大型語言模型幻覺，並讓回答更符合不同食品法規問題情境。

### 2.9 前端介面呈現

作法：

1. 使用 Streamlit 建立網頁介面。
2. 頁面標題為「台灣食品法規問答系統」。
3. 使用 `st.chat_input()` 接收使用者問題。
4. 使用 `st.chat_message()` 呈現使用者與系統的對話。
5. 使用 `st.session_state.history` 保存目前工作階段的聊天紀錄。
6. 回答完成後，使用 `st.expander()` 顯示參考來源。
7. 參考來源會分成正式法規、手冊 / 問答、裁罰案例與其他資料。
8. 每筆來源會顯示來源檔案、來源分類、檢索類型、相似度與部分文字內容。

目的：

提供直覺式聊天介面，讓使用者不需要理解資料庫或模型操作，也能查詢食品法規並看到回答依據。

### 2.10 部署與設定管理

作法：

1. 本地開發可使用 `.env` 儲存 API key。
2. Streamlit Cloud 部署時使用 Streamlit secrets 儲存 API key。
3. `app.py` 的 `_secret()` 函式會優先讀取環境變數，再讀取 `st.secrets`。
4. 雲端部署只需安裝 `requirements.txt`，不需要安裝本地建索引用的重型套件。
5. 本地建立索引時才安裝 `requirements_local.txt`，其中包含 `FlagEmbedding`、`torch`、`transformers` 等。
6. 部署平台建議使用 Streamlit Community Cloud，主程式入口為 `app.py`。

目的：

將「線上問答」與「本地建索引」的環境需求分開，讓雲端部署更輕量，也避免在 Streamlit Cloud 上安裝過大的模型與 GPU 相關套件。

## 3. 使用技術與工具

| 類別 | 工具 / 技術 | 用途 |
|---|---|---|
| 程式語言 | Python | 系統主要開發語言 |
| Web UI | Streamlit | 建立互動式問答網頁與聊天介面 |
| RAG 架構 | Retrieval-Augmented Generation | 結合法規檢索與生成式回答 |
| 向量資料庫 | Pinecone | 儲存與查詢法規段落 embedding |
| Embedding | BAAI/bge-m3 | 將問題與文件段落轉為向量 |
| Embedding API | Hugging Face Inference API | 雲端部署時產生 embedding |
| 本地 Embedding | FlagEmbedding | 本地建立索引與測試 embedding |
| 生成模型 API | Google Generative AI SDK | 呼叫 Gemini 生成回答 |
| 資料處理 | pypdf、python-docx | 補處理 PDF / DOCX 文字內容 |
| 數值處理 | NumPy | 向量格式轉換與 L2 normalize |
| 設定管理 | python-dotenv、Streamlit secrets | 管理 API key 與部署設定 |
| 進度與批次 | tqdm、checkpoint JSON | 建立索引時顯示進度與支援斷點續傳 |
| 部署平台 | Streamlit Community Cloud | 建議的雲端部署平台 |
| 版本控制 | Git / GitHub | 程式碼管理與部署來源 |

## 4. 使用模型

| 模型 | 類型 | 使用位置 | 說明 |
|---|---|---|---|
| `BAAI/bge-m3` | Embedding model | `app.py`、`build_index.py`、`query_test.py`、`supplemental_upsert_hf.py` | 將法規段落與使用者問題轉為 1024 維向量，用於語意檢索 |
| `gemini-3.5-flash` | 生成式語言模型 | `app.py` | 根據檢索到的法規與案例內容產生繁體中文回答 |
| `chi_tra` / `eng` | OCR 語言資料 | 前處理紀錄 | 用於繁體中文與英文 OCR 補萃取 |

備註：目前程式碼實際設定的生成模型為 `gemini-3.5-flash`。`PROGRESS.md` 的歷史紀錄曾提到 Gemini 1.5 Flash，若製作正式報告建議以 `app.py` 的實際設定為準。

## 5. 資料來源與知識庫

本系統知識庫主要包含：

- 台北市政府公告 114 至 115 年食品 / 健康食品違規廣告處罰案件統計表
- 食藥署（TFDA）食品安全衛生管理法相關條款
- GHP、HACCP 指引與相關問答集
- 食品檢驗方法、食品添加物、農藥殘留、動物用藥殘留、重金屬、微生物、標示與廣告等法規文件

資料前處理紀錄顯示，原始文件約 1,010 份，包含 PDF、DOCX、DOC 與 TXT。經文字萃取、OCR 與補處理後，已建立可供檢索的文字資料，並上傳至 Pinecone `food-rag` index。

## 6. 主要程式模組

| 檔案 | 功能 |
|---|---|
| `app.py` | Streamlit 問答系統主程式，負責 UI、檢索與 Gemini 回答生成 |
| `build_index.py` | 將 `extracted_texts/` 的文字檔切段、embedding 並 upsert 到 Pinecone |
| `query_test.py` | 測試 Pinecone 檢索結果是否正確 |
| `supplemental_process.py` | 補處理文字萃取異常或警告的原始文件 |
| `supplemental_upsert_hf.py` | 使用 Hugging Face Inference API 將補處理資料重新 embedding 並 upsert |
| `check_gpu.py` | 檢查本地 GPU / CUDA 是否可用 |
| `requirements.txt` | 雲端部署必要套件 |
| `requirements_local.txt` | 本地建索引與測試所需額外套件 |
| `DEPLOY.md` | Streamlit Cloud 部署說明 |
| `PROGRESS.md` | 專案資料處理與建置進度紀錄 |

## 7. 系統功能

1. 食品法規自然語言問答
2. 食品廣告、標示、裁罰案例查詢
3. 法規導向檢索與案例相似檢索合併
4. 第二階段 reranking 重新排序候選片段
5. 回答時依來源分類顯示參考來源、相似度與重排分數
6. 支援本地開發與雲端部署
7. 支援索引建立斷點續傳
8. 支援補處理資料重新上傳向量資料庫
9. 依問題類型切換回答格式
10. 以 `eval_questions.json` 建立代表性測試問題集

## 8. 環境需求

必要環境變數：

```env
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
```

雲端執行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

本地建立索引：

```bash
pip install -r requirements_local.txt
python build_index.py
```

## 9. 部署規劃

建議部署於 Streamlit Community Cloud：

- Repository：`garylin0909/FinalTerm_RAG`
- Branch：`main`
- Main file path：`app.py`
- 依賴套件：`requirements.txt`
- Secrets：`PINECONE_API_KEY`、`GEMINI_API_KEY`、`HF_TOKEN`

部署時不需要上傳原始 `data/` 與 `extracted_texts/`，前提是 Pinecone index 已完成建立並可正常查詢。

## 10. 後續優化方向

1. 將目前規則式 reranking 升級為 cross-encoder 或專用 reranker 模型，提高檢索段落的精準度。
2. 建立自動化評測腳本，讀取 `eval_questions.json` 並檢查拒答、來源分類與回答格式。
3. 建立管理介面，讓使用者可更新法規資料並重新建立索引。
4. 強化 Pinecone metadata，例如 `doc_type`、`topic`、`year` 與 `agency`，降低靠檔名猜分類的比例。
5. 統一模型文件紀錄，避免 README、PROGRESS 與程式碼出現模型版本不一致。
