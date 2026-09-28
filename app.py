import streamlit as st
from datetime import datetime
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rag_engine import LegalRAGEngine
import fitz
import speech_recognition as sr
import pyttsx3
from fpdf import FPDF
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Nyaya AI — Legal Assistant", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(20, 24, 38, 1) 0%, rgba(11, 13, 19, 1) 90.2%);
}

.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    padding: 4px 8px;
    margin-bottom: 10px;
}

.stButton button {
    border-radius: 10px;
    border: 1px solid rgba(212,175,55,0.35);
    transition: all 0.2s ease;
}
.stButton button:hover {
    border-color: #D4AF37;
    color: #D4AF37;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14171F 0%, #0E1117 100%);
    border-right: 1px solid rgba(212,175,55,0.1);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(255, 255, 255, 0.03);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.15);
}
.stTabs [data-baseweb="tab"] {
    height: 45px;
    border-radius: 8px;
    color: #9CA3AF;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #D4AF37 0%, #AA7C11 100%) !important;
    color: #000000 !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
    color: #F4E4BC !important;
    font-size: 32px !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(212,175,55,0.05) 100%) !important;
    border: 1px solid rgba(212,175,55,0.25) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

hr { border-color: rgba(212,175,55,0.2) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="font-family: 'Playfair Display', serif; background: linear-gradient(90deg, #D4AF37, #F4E4BC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 44px; margin-bottom: 0;">
        ⚖️ Nyaya AI / Legal AI
    </h1>
    <p style="color: #9CA3AF; font-size: 14px; margin-top: 4px; letter-spacing: 0.3px;">
        Powered by Hybrid RAG + 4 Trained Deep Learning Models · Constitution of India · Consumer Protection Act, 2019
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ System Configuration")
selected_model = st.sidebar.selectbox("Local LLM Engine (Ollama):", ["llama3.2:1b", "llama3.2:3b"], index=0)
selected_language = st.sidebar.selectbox("🌐 Answer Language:", ["English", "Hindi"], index=0)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📄 Upload a legal PDF (session-only)", type=["pdf"])

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []
if "shown_images" not in st.session_state:
    st.session_state.shown_images = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


@st.cache_resource
def get_rag_engine(model_name: str):
    return LegalRAGEngine(model_name=model_name)


try:
    engine = get_rag_engine(selected_model)
    st.sidebar.success("Engine initialized & linked to local vector store!")
except Exception as e:
    st.sidebar.error("Error loading engine. Have you run `ingest.py`?")
    st.stop()

if uploaded_file is not None and st.sidebar.button("Ingest this document"):
    with st.spinner("Reading and embedding your document..."):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        st.session_state.uploaded_chunks = engine.embed_and_chunk_text(full_text)
    st.sidebar.success(f"Ingested {len(st.session_state.uploaded_chunks)} chunks from '{uploaded_file.name}'!")

if st.session_state.uploaded_chunks:
    st.sidebar.info(f"📎 {len(st.session_state.uploaded_chunks)} chunks from your uploaded document are active.")

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Document Image Classifier")
uploaded_image = st.sidebar.file_uploader("Upload a page image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="image_uploader")

if uploaded_image is not None:
    st.sidebar.image(uploaded_image, caption="Uploaded page", use_container_width=True)
    if st.sidebar.button("Classify this image"):
        with st.spinner("Running CNN classification..."):
            image_bytes = uploaded_image.read()
            result = engine.classify_document_image(image_bytes)
        st.sidebar.success(f"Predicted: **{result['predicted_class']}** ({result['confidence']}% confidence)")
        st.sidebar.write(result["all_scores"])

if st.session_state.messages:
    export_text = f"# Legal AI Assistant — Conversation Export\n_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
    for m in st.session_state.messages:
        role = "**You**" if m["role"] == "user" else "**Assistant**"
        export_text += f"{role}: {m['content']}\n\n---\n\n"
    st.sidebar.markdown("---")
    st.sidebar.download_button("⬇️ Export Conversation (.md)", export_text, file_name="legal_ai_chat.md", mime="text/markdown")


def transcribe_audio(audio_file) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return ""


def text_to_speech_bytes(text: str) -> bytes:
    temp_path = "temp_tts_output.wav"
    try:
        tts_engine = pyttsx3.init()
        tts_engine.save_to_file(text, temp_path)
        tts_engine.runAndWait()
        with open(temp_path, "rb") as f:
            data = f.read()
        os.remove(temp_path)
        return data
    except Exception:
        return b""


def generate_pdf_bytes(question: str, answer: str, timestamp: str) -> bytes:
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Nyaya AI - Legal Answer Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {timestamp}", ln=True)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, f"Question: {question}")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, f"Answer: {answer}")
        return bytes(pdf.output(dest="S"))
    except Exception:
        return b""


def get_pdf_page_image(pdf_path: str, page_num: int) -> bytes:
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=120)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None


user_query = st.chat_input("Ask a legal question based on your indexed normative documents...")
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

tab_chat, tab_dashboard, tab_about, tab_graph = st.tabs(["💬 Chat", "📊 Analytics Dashboard", "ℹ️ About This Project", "🕸️ Knowledge Graph"])

with tab_chat:
    with st.expander("🎤 Ask by voice"):
        audio_input = st.audio_input("Record your question")
        if audio_input is not None:
            audio_id = audio_input.file_id if hasattr(audio_input, "file_id") else str(len(audio_input.getvalue()))
            if audio_id != st.session_state.last_audio_id:
                st.session_state.last_audio_id = audio_id
                with st.spinner("Transcribing your voice..."):
                    transcribed = transcribe_audio(audio_input)
                if transcribed:
                    st.success(f"Heard: \"{transcribed}\"")
                    st.session_state.pending_query = transcribed
                    st.rerun()
                else:
                    st.warning("Could not understand the audio. Please try again or type your question.")

    if not st.session_state.messages:
        st.markdown("<p style='color:#9CA3AF; font-size:14px; margin-bottom:6px;'>Try asking:</p>", unsafe_allow_html=True)
        sample_qs = [
            "What are the six consumer rights under the Consumer Protection Act?",
            "What does Article 21 of the Constitution protect?",
            "Explain the writ jurisdiction under Article 32."
        ]
        chip_cols = st.columns(len(sample_qs))
        for i, q in enumerate(sample_qs):
            if chip_cols[i].button(q, key=f"sample_chip_{i}"):
                st.session_state.pending_query = q
                st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if "min_distance" in msg and msg["min_distance"] is not None:
                min_dist = msg["min_distance"]
                if min_dist < 0.8:
                    st.markdown("🟢 **Confidence: High**")
                elif min_dist < 1.1:
                    st.markdown("🟡 **Confidence: Medium**")
                else:
                    st.markdown("🟠 **Confidence: Low**")

            if "groundedness" in msg and msg["groundedness"] is not None:
                g_score = msg["groundedness"]
                if g_score >= 70:
                    st.markdown(f"✅ **Groundedness: {g_score:.1f}%** — well-supported by retrieved context")
                elif g_score >= 45:
                    st.markdown(f"⚠️ **Groundedness: {g_score:.1f}%** — partially supported, verify key claims")
                else:
                    st.markdown(f"🚫 **Groundedness: {g_score:.1f}%** — low support, possible hallucination")


            if msg.get("topic_image"):
                st.image(msg["topic_image"], caption=f"Related topic: {msg.get('predicted_domain', '').replace('_', ' ').title()}", width=350)

            if msg["role"] == "assistant":
                action_cols = st.columns([1, 1, 6])
                if action_cols[0].button("👍", key=f"up_{idx}"):
                    msg["feedback"] = "up"
                    st.rerun()
                if action_cols[1].button("👎", key=f"down_{idx}"):
                    msg["feedback"] = "down"
                    st.rerun()

                if msg.get("feedback"):
                    st.caption(f"Feedback recorded: {'👍 Helpful' if msg['feedback'] == 'up' else '👎 Not helpful'}")

                if st.button("🔊 Listen to this answer", key=f"listen_{idx}"):
                    with st.spinner("Generating speech..."):
                        audio_bytes = text_to_speech_bytes(msg["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                    else:
                        st.warning("Could not generate audio for this answer.")

                pdf_bytes = generate_pdf_bytes(
                    st.session_state.messages[idx - 1]["content"] if idx > 0 else "N/A",
                    msg["content"],
                    datetime.now().strftime('%Y-%m-%d %H:%M')
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 Download this answer as PDF",
                        pdf_bytes,
                        file_name=f"legal_answer_{idx}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}",
                        use_container_width=True
                    )

            if "followups" in msg and msg["followups"]:
                st.markdown("---")
                st.caption("💡 **Suggested Follow-up Questions:**")
                cols = st.columns(len(msg["followups"]))
                for f_idx, q in enumerate(msg["followups"]):
                    if cols[f_idx].button(q, key=f"btn_{idx}_{f_idx}"):
                        st.session_state.pending_query = q
                        st.rerun()

    if user_query:
        start_time = time.time()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching normative texts using Hybrid BM25 + Vector RRF..."):
                contexts, is_confident, min_distance, predicted_domain = engine.retrieve(user_query)
                uploaded_hits = engine.search_uploaded(user_query, st.session_state.uploaded_chunks, top_n=2)
                if uploaded_hits:
                    contexts = uploaded_hits + contexts
                    is_confident = True

            if contexts:
                with st.sidebar.expander("🔍 Retrieved Citation Sources & Visuals", expanded=True):
                    for c_idx, ctx in enumerate(contexts):
                        meta = ctx.get("metadata", {})
                        st.markdown(f"**Source {c_idx+1}:** `{meta.get('source', 'Unknown')}`")
                        st.markdown(f"**Citation:** `{meta.get('citation', 'N/A')}` (Page {meta.get('page', '-')})")
                        text_snippet = ctx.get("text", "")[:180]
                        st.caption(f"{text_snippet}...")

                        source_name = meta.get("source", "")
                        page_val = meta.get("page", "-")
                        if source_name and page_val != "-":
                            src_path = os.path.join("data", source_name)
                            try:
                                page_num = int(page_val)
                                if os.path.exists(src_path):
                                    thumb = get_pdf_page_image(src_path, page_num)
                                    if thumb:
                                        st.image(thumb, caption=f"Page {page_num} preview", use_container_width=True)
                            except (ValueError, TypeError):
                                pass
                        st.divider()

            response_container = st.empty()
            full_response = ""

            for token in engine.generate_stream(user_query, contexts, language=selected_language):
                full_response += token
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)

            topic_image_path = engine.get_topic_image(predicted_domain, st.session_state.shown_images)
            if topic_image_path:
                st.session_state.shown_images.add(topic_image_path)
                st.image(topic_image_path, caption=f"Related topic: {predicted_domain.replace('_', ' ').title()}", width=350)

            with st.spinner("Checking groundedness..."):
                groundedness = engine.check_groundedness(full_response, contexts)

            answer_metrics = engine.compute_answer_metrics(full_response, contexts)

            with st.spinner("Generating follow-up suggestions..."):
                followups = engine.generate_followups(user_query, full_response)

            elapsed = round(time.time() - start_time, 2)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "min_distance": min_distance,
                "groundedness": groundedness,
                "followups": followups,
                "topic_image": topic_image_path,
                "predicted_domain": predicted_domain,
                "answer_metrics": answer_metrics
            })

            st.session_state.query_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": user_query[:60],
                "domain": predicted_domain,
                "confidence_distance": round(min_distance, 3),
                "groundedness": groundedness,
                "response_time_sec": elapsed
            })

        st.rerun()

with tab_dashboard:
    st.subheader("📊 System Analytics")

    if not st.session_state.query_log:
        st.info("Ask a few questions in the Chat tab to see analytics here.")
    else:
        log_df = pd.DataFrame(st.session_state.query_log)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", len(log_df))
        col2.metric("Avg Groundedness", f"{log_df['groundedness'].mean():.1f}%")
        col3.metric("Avg Response Time", f"{log_df['response_time_sec'].mean():.2f}s")

        feedback_counts = {"up": 0, "down": 0}
        for m in st.session_state.messages:
            if m.get("feedback") == "up":
                feedback_counts["up"] += 1
            elif m.get("feedback") == "down":
                feedback_counts["down"] += 1
        col4.metric("User Feedback", f"👍 {feedback_counts['up']} / 👎 {feedback_counts['down']}")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=log_df['groundedness'].mean(),
                title={'text': "Average Groundedness (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#D4AF37"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            fig_pie = px.pie(
                log_df,
                names='domain',
                title='Query Domain Distribution',
                hole=0.5,
                color_discrete_sequence=['#D4AF37', '#2563EB', '#10B981']
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        fig_scatter = px.scatter(
            log_df,
            x='response_time_sec',
            y='groundedness',
            color='domain',
            size='confidence_distance',
            hover_data=['query'],
            title="Response Time vs. Groundedness"
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Full Query Log")
        st.dataframe(log_df, use_container_width=True)

with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
**Nyaya AI** is a Retrieval-Augmented Generation (RAG) legal assistant built for the Constitution of India and the Consumer Protection Act, 2019 — extending the methodology of *"Legal AI for All"* (García-Montero et al., IEEE Access, 2025) with four trained deep learning components.

#### Pipeline
1. **Query Domain Classifier** (self-trained Logistic Regression on bge-m3 embeddings) — rejects out-of-domain questions
2. **Hybrid Retrieval** — BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion
3. **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) — re-scores retrieved chunks for true relevance
4. **Local LLM Generation** (Ollama, llama3.2) — generates the grounded answer
5. **NLI Groundedness Scorer** (`nli-deberta-v3-small`) — verifies the answer is actually supported by the retrieved text
6. **CNN Document Image Classifier** (MobileNetV2, transfer learning) — a standalone image-based classifier trained on a custom dataset

#### Key differences from the original paper
- Domain: Indian law instead of Ecuadorian law
- No fine-tuning (hardware constraint) — replaced with 3 additional trained/pretrained DL models
- A full interactive application instead of an offline research evaluation
- Bilingual (English/Hindi) answers, voice input/output, and PDF export

#### Benchmark result
Retrieval accuracy improved from **85.7% to 100%** after adding the query domain classifier as a second guardrail layer, measured on a custom difficulty-tiered benchmark.
""")

    st.markdown("#### 📐 Formal Evaluation Metrics")
    st.caption("Measured via cross-validation (classifier), held-out validation split (CNN), and a 16-question labeled test set (guardrail).")

    eval_data = {
        "Model": ["Query Domain Classifier", "Document Image Classifier (CNN)", "Retrieval Guardrail"],
        "Accuracy": ["86.7%", "85.9%", "100.0%*"],
        "Precision (macro)": ["86.6%", "87.9%", "100.0%*"],
        "Recall (macro)": ["86.7%", "88.7%", "100.0%*"],
        "F1-score (macro)": ["86.6%", "85.8%", "100.0%*"],
    }
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    st.caption("*Guardrail evaluated on a small (16-example) labeled test set — a larger adversarial test set would be needed for a stronger production claim.")


with tab_graph:
    st.subheader("🕸️ Legal Knowledge Graph")
    st.caption("Illustrative diagram showing how the indexed legal domains relate. Not dynamically generated from live queries.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image("static_knowledge_graph.png", caption="Static Legal Domain Map", use_container_width=True)
    with img_col2:
        st.image("static_pipeline_diagram.png", caption="Processing Pipeline Overview", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Interactive Version")

    nodes = [
        Node(id="Indian Legal System", label="Indian Legal System", color="#D4AF37", size=25),
        Node(id="Constitution", label="Constitution of India", color="#2563EB", size=20),
        Node(id="Consumer Act", label="Consumer Protection Act 2019", color="#10B981", size=20),
        Node(id="Art 14", label="Article 14 (Equality)", color="#9333EA", size=15),
        Node(id="Art 21", label="Article 21 (Life & Liberty)", color="#9333EA", size=15),
        Node(id="Art 32", label="Article 32 (Writs)", color="#9333EA", size=15),
        Node(id="Art 246", label="Article 246 (Legislative Powers)", color="#9333EA", size=15),
        Node(id="Consumer Rights", label="Six Consumer Rights", color="#F59E0B", size=15),
        Node(id="Deficiency", label="Deficiency of Service", color="#F59E0B", size=15),
        Node(id="District Comm", label="District Commission", color="#F59E0B", size=15),
    ]
    edges = [
        Edge(source="Indian Legal System", target="Constitution"),
        Edge(source="Indian Legal System", target="Consumer Act"),
        Edge(source="Constitution", target="Art 14"),
        Edge(source="Constitution", target="Art 21"),
        Edge(source="Constitution", target="Art 32"),
        Edge(source="Constitution", target="Art 246"),
        Edge(source="Consumer Act", target="Consumer Rights"),
        Edge(source="Consumer Act", target="Deficiency"),
        Edge(source="Consumer Act", target="District Comm"),
    ]

    config = Config(width=900, height=500, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)
