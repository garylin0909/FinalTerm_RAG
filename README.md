# Food Regulation RAG Q&A

Streamlit web app for food regulation question answering.

The app uses:

- Hugging Face Inference API for `BAAI/bge-m3` embeddings in cloud deployment
- Pinecone for vector search
- Google Gemini for answer generation
- Streamlit for the web UI

## Run Locally

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
```

Install the cloud app dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local index building, install the heavier local dependencies:

```bash
pip install -r requirements_local.txt
python build_index.py
```

## Deploy

Recommended platform: Streamlit Community Cloud.

Use these settings:

- Repository: `garylin0909/FinalTerm_RAG`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`

Add these secrets in Streamlit Cloud:

```toml
PINECONE_API_KEY = "your_pinecone_key"
GEMINI_API_KEY = "your_gemini_key"
HF_TOKEN = "your_huggingface_token"
```

The Hugging Face token must include permission to make Inference Providers calls.

Do not upload `.env` or `.streamlit/secrets.toml`.
