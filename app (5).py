import streamlit as st
import time
from pypdf import PdfReader
import os
import sys
import html

# Ensure local package directory is on sys.path so imports from the local RAG package work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from RAG.embeddings import get_jina_embeddings
from RAG.vision import describe_image
from RAG.chunking import chunk_text
from RAG.retriever import FAISSRetriever
from RAG.reranker import simple_rerank
from RAG.llm import ask_llm


st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🤖", layout="wide")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@600;700&display=swap');
:root{
  --bg-1: #0f172a;
  --bg-2: #0b1222;
  --glass: rgba(255,255,255,0.06);
  --muted: #9aa4b2;
  --card: rgba(255,255,255,0.02);
  --accent: #60a5fa;
  --accent-2: #7c3aed;
  --radius: 14px;
}
body, #root, .appview-container {
  background: radial-gradient(900px 400px at 10% 10%, #0b122933, transparent 25%),
              radial-gradient(700px 300px at 92% 15%, #3b82f633, transparent 20%),
              linear-gradient(180deg,#071126,#081228 60%);
  color: #e6eef8;
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.nav{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:12px;margin-bottom:18px;border:1px solid rgba(255,255,255,0.03)}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;font-weight:700;color:white}
.title{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700}
.subtitle{color:var(--muted);font-size:12px}
.nav-actions{display:flex;gap:10px;align-items:center}
.btn{background:linear-gradient(90deg,var(--accent),var(--accent-2));padding:8px 12px;border-radius:10px;border:0;color:#011; font-weight:600;cursor:pointer;box-shadow:0 8px 30px rgba(92, 70, 255, 0.12);transition:transform .12s ease}
.btn:hover{transform:translateY(-3px)}

/* Style Streamlit buttons and file uploader controls (use app accents) */
.stButton>button,
.stFileUploader button,
.stFileUploader div > button {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 0 !important;
    padding: 8px 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 30px rgba(92, 70, 255, 0.12) !important;
    transition: transform .12s ease, filter .12s ease !important;
}
.stButton>button:hover,
.stFileUploader button:hover,
.stFileUploader div > button:hover {
    transform: translateY(-3px) !important;
    filter: brightness(1.05) !important;
}

.container{max-width:1100px;margin:0 auto}
.main-grid{display:grid;grid-template-columns: 1fr 360px;gap:20px}
.panel{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:var(--radius);padding:18px;border:1px solid rgba(255,255,255,0.03);box-shadow:0 10px 30px rgba(2,6,23,0.6)}
.uploader{display:flex;flex-direction:column;gap:10px}
.status-row{display:flex;gap:12px;margin-top:10px}
.status-pill{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,0.03);color:var(--muted);font-weight:600;font-size:13px}

.chat-area{height:60vh;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:12px}
.bubble{max-width:78%;padding:12px 14px;border-radius:14px;line-height:1.45}
.bubble.user{background:linear-gradient(90deg,#0ea5a5,#06b6d4);margin-left:auto;border-bottom-right-radius:4px;color:#021}
.bubble.bot{background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));border-bottom-left-radius:4px;color:#e6eef8}
.input-row{display:flex;gap:10px;align-items:center;padding-top:8px}
.query-input{flex:1;padding:12px 14px;border-radius:12px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03);color:inherit}
.small{font-size:12px;color:var(--muted)}

/* typing */
.typing{display:inline-block;height:12px}
.typing span{display:inline-block;width:6px;height:6px;background:#94a3b8;border-radius:50%;margin-right:6px;animation:blink 1s infinite}
.typing span:nth-child(2){animation-delay:.15s}
.typing span:nth-child(3){animation-delay:.3s}
@keyframes blink{0%{opacity:.2}50%{opacity:1}100%{opacity:.2}}

/* responsive */
@media (max-width:900px){.main-grid{grid-template-columns:1fr}.panel{padding:12px}}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _ensure_history():
    if "history" not in st.session_state:
        st.session_state.history = []


_ensure_history()


# Top navigation
st.markdown(
    """
    <div class='nav container'>
      <div class='brand'>
        <div class='logo'>AI</div>
        <div>
          <div class='title'>Multimodal RAG</div>
          <div class='subtitle'>Grounded answers for your documents & images</div>
        </div>
      </div>
      <div class='nav-actions'>
        <button class='btn'>Examples</button>
        <button class='btn'>Docs</button>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Layout: main + sidebar
left, right = st.columns([1, 0.38], gap="large")

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "openai/gpt-oss-120b"])
    filter_type = st.radio("Retrieval Scope", ["all", "text", "image"], horizontal=True)
    st.markdown("<div class='small'>Tip: Use <strong>image</strong> scope to test visual grounding.</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='small'>Recent Uploads</div>", unsafe_allow_html=True)
    if "last_upload" in st.session_state:
        st.markdown(f"<div class='small'>• {st.session_state['last_upload']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small'>No uploads yet</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='main-grid'>", unsafe_allow_html=True)

    # Main column
    main_col, side_col = st.columns([2, 0.0001])
    with main_col:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 8px 0'>Document & Image Upload</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            txt_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"], key="doc_uploader")
            if txt_file:
                st.success(f"Loaded: {txt_file.name}")
                st.session_state['last_upload'] = txt_file.name
        with col2:
            img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"], key="img_uploader")
            if img_file:
                st.success(f"Loaded: {img_file.name}")
                st.session_state['last_upload'] = img_file.name

        st.markdown("<div class='status-row'>", unsafe_allow_html=True)
        st.markdown("<div class='status-pill'>Document: " + (txt_file.name if txt_file else "Pending") + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='status-pill'>Image: " + (img_file.name if img_file else "Optional") + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Chat panel
        st.markdown("<div class='panel' style='margin-top:16px'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 8px 0'>Chat</h3>", unsafe_allow_html=True)
        chat_area = st.container()
        with chat_area:
            chat_box = st.empty()
            def render_chat():
                html_parts = ["<div class='chat-area'>"]
                for q, a in st.session_state.history:
                    html_parts.append(f"<div class='bubble user'>{html.escape(q)}</div>")
                    html_parts.append(f"<div class='bubble bot'>{html.escape(a)}</div>")
                html_parts.append("</div>")
                chat_box.markdown("".join(html_parts), unsafe_allow_html=True)

            render_chat()

        # Input form
        st.markdown("<div class='input-row'>", unsafe_allow_html=True)
        query = st.text_input("Question", placeholder="Ask a question about the uploaded content...", key="query_input", label_visibility="collapsed")
        run = st.button("Send")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Helper: show typing indicator
def _typing_placeholder():
    t = st.empty()
    t.markdown("<div class='typing'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
    return t


system_ready = bool(txt_file and groq_key and jina_key)

if not system_ready:
    st.warning("Provide document and API keys to enable retrieval.")


if txt_file and groq_key and jina_key:

    # Process uploads only once — keep logic identical but show progress
    if "processed" not in st.session_state or st.session_state.get("doc_name") != getattr(txt_file, "name", None):
        processing = st.empty()
        with processing.container():
            st.info("Processing knowledge sources…")
            progress = st.progress(0)
            time.sleep(0.15)
            progress.progress(20)

            if txt_file.name.endswith('.pdf'):
                reader = PdfReader(txt_file)
                raw_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            else:
                raw_text = txt_file.read().decode('utf-8')

            progress.progress(50)
            chunks = chunk_text(raw_text)
            metadata = [{"type": "text"} for _ in chunks]

            if img_file:
                image_bytes = img_file.read()
                try:
                    vision_text = describe_image(image_bytes, groq_key)
                except Exception as e:
                    # describe_image may raise an authentication error from the
                    # Groq client or a generic network/error. Surface a friendly
                    # message without assuming the `groq` module is present.
                    msg = str(e).lower()
                    if 'authentication' in msg or '401' in msg or 'unauthorized' in msg:
                        st.error("Authentication failed for Groq API. Please verify your Groq API key in Configuration.")
                    else:
                        st.error("Image understanding failed: " + str(e))
                    vision_text = None

                if vision_text:
                    chunks.append("Image description: " + vision_text)
                    metadata.append({"type": "image"})

            progress.progress(75)
            try:
                embeddings = get_jina_embeddings(chunks, jina_key)
            except Exception as e:
                # Catch HTTP/auth errors from Jina and show a friendly message
                st.error("Failed to create embeddings: " + str(e))
                processing.error("Embedding failure — check your Jina API key and network.")
                st.stop()

            retriever = FAISSRetriever(embeddings, metadata)

            st.session_state['chunks'] = chunks
            st.session_state['metadata'] = metadata
            st.session_state['retriever'] = retriever
            st.session_state['processed'] = True
            st.session_state['doc_name'] = txt_file.name

            progress.progress(100)
            processing.success("Processing complete")
            time.sleep(0.4)

    if run and query:
        # show typing
        typing = _typing_placeholder()
        start = time.time()

        try:
            query_emb = get_jina_embeddings([query], jina_key)
        except Exception as e:
            st.error("Failed to compute query embedding: " + str(e))
            typing.empty()
            st.stop()
        f = None if filter_type == 'all' else filter_type
        # Retrieve more chunks (top 10) to ensure relevant content is found
        ids = st.session_state['retriever'].search(query_emb, top_k=10, filter_type=f)
        retrieved_docs = [st.session_state['chunks'][i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)
        # Use top 5 reranked chunks; if empty, fall back to retrieved chunks
        selected_chunks = reranked[:5] if reranked else retrieved_docs[:5]
        context = "\n\n".join(selected_chunks)
        
        # Warn if context is very sparse
        if not context or len(context.strip()) < 100:
            st.warning("Limited relevant context found. Results may be incomplete. Try a different question or check your document.")

        try:
            answer = ask_llm(context, query, groq_key, model)
        except Exception as e:
            error_msg = str(e).lower()
            if 'connection' in error_msg or 'network' in error_msg or 'timeout' in error_msg:
                st.error("❌ Connection Error: Unable to reach Groq API. Please check:\n- Your internet connection\n- Groq API status (groq.com)\n- Your firewall/proxy settings\n- Try again in a moment.")
            elif 'authentication' in error_msg or 'invalid' in error_msg or '401' in error_msg:
                st.error("❌ Authentication Failed: Check your Groq API key in the Configuration panel.")
            elif '429' in error_msg or 'rate' in error_msg:
                st.error("❌ Rate Limit: Too many requests. Please wait a moment and try again.")
            else:
                st.error(f"❌ LLM Error: {str(e)[:200]}")
            answer = f"Error: Unable to generate answer. {str(e)[:100]}"
        latency = round(time.time() - start, 2)

        # append history and re-render
        st.session_state.history.append((query, answer))
        typing.empty()

        # update chat area
        # reuse render logic
        chat_html = ["<div class='chat-area'>"]
        for q, a in st.session_state.history:
            chat_html.append(f"<div class='bubble user'>{html.escape(q)}</div>")
            chat_html.append(f"<div class='bubble bot'>{html.escape(a)}</div>")
        chat_html.append("</div>")
        st.markdown("".join(chat_html), unsafe_allow_html=True)

        st.metric("Latency (s)", latency)
        with st.expander("Retrieved Context"):
            st.text(context)

        with st.expander("Recent Chat History"):
            for q, a in st.session_state.history[-8:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.divider()
else:
    # keep the same informational panel when not ready
    st.info("Upload a document and provide API keys to begin.")

