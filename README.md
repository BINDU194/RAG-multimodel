# Enterprise Multimodal RAG (Text + Vision)

This project is an end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that enables question answering over both:

# Enterprise Multimodal RAG (Text + Vision)

This repository provides a local Streamlit application for multimodal Retrieval-Augmented Generation (RAG):

- Ingest text documents (TXT, PDF) and images (PNG/JPG)
- Create embeddings with Jina Embeddings v4
- Retrieve relevant chunks with FAISS and a small reranker
- Generate grounded answers using Groq LLMs and Groq Vision for images

---

## Installation & Run (Quick)

1. Create a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

2. Install dependencies:

```powershell
pip install -r "requirements (2).txt"
```

3. Start the app:

```powershell
streamlit run "app (5).py"
```

4. Open the UI in your browser at:

```
http://localhost:8501
```

---

## API Keys & Configuration

- Groq LLM & Vision: create an API key at https://console.groq.com and paste it into the **Groq API Key** field in the app sidebar.
- Jina Embeddings: get an API key from Jina (https://jina.ai) and paste it into the **Jina API Key** field.

The app keeps keys in session memory only (they are not persisted to disk).

---

## Usage (Short)

1. Open the app and enter your Groq and Jina API keys in the Configuration panel.
2. Upload a TXT or PDF file using the "Upload TXT or PDF" control.
3. (Optional) Upload an image (PNG/JPG) for multimodal grounding.
4. Ask a question in the chat input and press Send.

The app will:

- Chunk and embed the document text
- Optionally describe the image via Groq Vision and include that text in the context
- Retrieve and rerank top-k chunks from FAISS
- Call Groq LLM with the retrieved context and return a grounded answer

---

## Troubleshooting — Common Issues

- "Unable to connect" to Streamlit UI:
  - Make sure port 8501 is available and not blocked by a firewall.
  - If port conflicts occur, run `streamlit run "app (5).py" --server.port 8504` and open `http://localhost:8504`.

- "Connection Error: Unable to reach Groq API":
  - Verify your internet connection.
  - Confirm your Groq API key is correct and active.
  - Check local firewall or corporate proxy settings that may block outbound HTTPS to Groq.

- Jina embedding errors (HTTP 401 / authentication):
  - Confirm your Jina API key is valid and has quota.

- Slow responses:
  - Network latency to external APIs (Groq, Jina) is common. Retry after a short wait.

If you hit issues you can't resolve, open an issue describing the failure and include screenshots and any terminal output.

---

## Developer Notes

- Main UI: `app (5).py`
- RAG helper modules: the `RAG/` package (`embeddings.py`, `llm.py`, `retriever.py`, `chunking.py`, `vision.py`, `reranker.py`).
- To run tests or quick checks, use the small `test_app.py` file or the `test_groq.py` diagnostic helper (if present).

---

If you'd like I can also add a short example question or a visual debug panel that shows retrieved chunks and rerank scores.
