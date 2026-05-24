# Deployment Guide

## Recommended: Streamlit Community Cloud

This project is a Streamlit app, so Streamlit Community Cloud is the simplest free hosting option.

## Before Deploying

Make sure these files are in GitHub:

- `app.py`
- `requirements.txt`
- `README.md`
- optional helper scripts such as `build_index.py`, `query_test.py`, and `check_gpu.py`

Make sure these files are not in GitHub:

- `.env`
- `.streamlit/secrets.toml`
- `.venv/`
- `data/`
- `extracted_texts/`
- checkpoint or generated report files

## Streamlit Cloud Settings

Create a new app at Streamlit Community Cloud and use:

- Repository: `garylin0909/FinalTerm_RAG`
- Branch: `main`
- Main file path: `app.py`

Streamlit Cloud will install dependencies from `requirements.txt`.

## Secrets

Add these values in Streamlit Cloud app secrets:

```toml
PINECONE_API_KEY = "your_pinecone_key"
GEMINI_API_KEY = "your_gemini_key"
HF_TOKEN = "your_huggingface_token"
```

For Hugging Face, create a token that can make Inference Providers calls.

The app reads environment variables first, then Streamlit secrets.

## Notes

The original PDF files and extracted text files are not required for the deployed app as long as the Pinecone index is already built and available.
