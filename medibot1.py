"""
MediBot - Medical Q&A Chatbot (Streamlit)
Production-style UI: trust signals, copy/regenerate, timestamps, dark mode.
"""
import streamlit as st
import time
from datetime import datetime

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="MediBot - Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Typography & layout constants
FONT_TITLE = "1.75rem"
FONT_BODY = "1rem"
FONT_SMALL = "0.875rem"
MAX_INPUT_CHARS = 2000


def inject_theme(dark_mode: bool):
    """Inject CSS: contrast, typography, fade-in, code blocks, dark mode."""
    sidebar_bg = "linear-gradient(180deg, #022c22 0%, #064e3b 50%, #065f46 100%)" if not dark_mode else "linear-gradient(180deg, #0f172a 0%, #1e293b 100%)"
    main_bg = "#ffffff" if not dark_mode else "#0f172a"
    chat_bg = "#f8fffe" if not dark_mode else "#1e293b"
    st.markdown(
        f"""
        <style>
        /* Typography: 3 sizes only */
        :root {{
            --font-title: {FONT_TITLE};
            --font-body: {FONT_BODY};
            --font-small: {FONT_SMALL};
        }}
        .medibot-hero h1 {{ font-size: var(--font-title) !important; }}
        .medibot-hero .tagline {{ font-size: var(--font-body) !important; }}
        .medibot-hero .badge {{ font-size: var(--font-small) !important; }}
        [data-testid="stChatMessage"] .stMarkdown {{ font-size: var(--font-body) !important; }}
        .msg-time {{ font-size: var(--font-small); opacity: 0.8; }}
        .trust-banner {{ font-size: var(--font-small); }}

        /* Main: lighter background for clear separation */
        .stApp {{ background: {main_bg}; }}
        .main .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; max-width: 900px; }}

        /* Sidebar: darker for contrast */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: 3px solid #064e3b;
        }}
        [data-testid="stSidebar"] .stMarkdown {{ color: #f0fdfa !important; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.25); }}

        /* Chat area: very light / distinct */
        .chat-scroll {{ max-height: calc(100vh - 320px); overflow-y: auto; padding-bottom: 1rem; margin-bottom: 0.5rem; }}
        [data-testid="stChatMessage"] {{
            animation: fadeIn 0.35s ease-out;
        }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* User: right, teal */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"]:only-of-type) + div [data-testid="stChatMessage"],
        .user-msg {{ margin-left: auto; margin-right: 0; max-width: 78%; }}
        div[data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"]) {{
            align-items: flex-end;
        }}
        /* Chat bubbles: base (assistant-style left) */
        [data-testid="stChatMessage"] {{
            padding: 0.75rem 1.1rem;
            border-radius: 18px;
            box-shadow: 0 2px 10px rgba(13, 148, 136, 0.15);
            max-width: 85%;
        }}
        /* User: teal, right-aligned via Streamlit default */
        [data-testid="stChatMessage"] .stMarkdown {{ font-size: var(--font-body) !important; }}
        /* Assistant: light mint (default) */
        div[data-testid="stChatMessage"] {{
            background: linear-gradient(135deg, #fff 0%, #f0fdfa 100%);
            color: #134e4a;
            border: 1px solid #99f6e4;
            border-radius: 18px 18px 18px 4px;
        }}
        div[data-testid="stChatMessage"] .stMarkdown {{ color: #134e4a !important; }}

        /* Code / pre for future reports */
        .stMarkdown pre {{ background: #0f766e; color: #ccfbf1; border-radius: 8px; padding: 0.75rem; font-size: var(--font-small); }}
        .stMarkdown code {{ background: #e0f2f1; color: #0f766e; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: var(--font-small); }}

        /* Input area */
        [data-testid="stChatInput"] {{ background: #fff; border-radius: 12px; border: 2px solid #0d9488; box-shadow: 0 2px 12px rgba(13,148,136,0.15); }}
        .stTextArea textarea {{ border: 2px solid #0d9488; border-radius: 12px; }}

        /* Typing indicator */
        @keyframes typing {{ 0%, 60%, 100% {{ opacity: 0.3; }} 30% {{ opacity: 1; }} }}
        .typing-indicator span {{ width: 8px; height: 8px; border-radius: 50%; background: #14b8a6; animation: typing 1.4s infinite; }}
        .typing-indicator span:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-indicator span:nth-child(3) {{ animation-delay: 0.4s; }}

        .med-strip {{ height: 4px; background: linear-gradient(90deg, #0d9488, #14b8a6); border-radius: 2px; margin-bottom: 0.75rem; }}
        .medibot-hero {{ background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); padding: 1.25rem 1.5rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 6px 20px rgba(13,148,136,0.3); }}
        .medibot-hero h1 {{ color: #fff !important; margin: 0 !important; }}
        .medibot-hero .tagline {{ color: rgba(255,255,255,0.95); margin-top: 0.35rem; }}
        .medibot-hero .badge {{ background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 20px; margin-top: 0.4rem; display: inline-block; }}

        /* Trust banner */
        .trust-banner {{ background: #fef3c7; color: #92400e; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 0.75rem; }}
        .confidence-high {{ color: #059669; }}
        .confidence-medium {{ color: #d97706; }}
        .confidence-low {{ color: #dc2626; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def msg_normalize(m):
    """Normalize message to dict with content, timestamp, sources, confidence, suggested_questions."""
    if isinstance(m, str):
        return {"role": "assistant", "content": m, "timestamp": None, "sources": None, "confidence": None, "suggested_questions": None}
    if isinstance(m, dict):
        return {
            "role": m.get("role", "assistant"),
            "content": m.get("content", ""),
            "timestamp": m.get("timestamp"),
            "sources": m.get("sources") or [],
            "confidence": m.get("confidence"),
            "suggested_questions": m.get("suggested_questions") or [],
        }
    return {"role": "assistant", "content": "", "timestamp": None, "sources": None, "confidence": None, "suggested_questions": None}


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 **Hello!** I'm **MediBot**, your medical Q&A assistant.\n\n"
                "Ask me questions and I'll answer using my **knowledge base** — try symptoms, conditions, treatments, or definitions.\n\n"
                "_Information only; always consult a healthcare professional for medical advice._",
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "confidence": None,
                "suggested_questions": ["What is diabetes?", "What are symptoms of anemia?", "How is hypertension treated?"],
            }
        ]
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🩺 MediBot")
        st.markdown("Medical Q&A from our knowledge base. _Not a substitute for professional care._")
        st.markdown("---")
        dark = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode, key="dark_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()
        st.markdown("#### 💡 Examples")
        st.markdown("- What is diabetes?\n- Symptoms of anemia?\n- How is hypertension treated?")
        st.markdown("---")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Ask me a medical question.",
                    "timestamp": datetime.now().isoformat(),
                    "sources": [],
                    "confidence": None,
                    "suggested_questions": [],
                }
            ]
            if st.session_state.get("is_processing"):
                st.session_state.is_processing = False
            st.rerun()


def get_bot_reply_placeholder(user_query: str):
    """Return placeholder reply with sources and confidence for demo."""
    return {
        "content": (
            "Thanks for your question. This is a **demo response**. In production I would answer from the indexed medical content.\n\n"
            "Example formatting:\n- **Bold** and _italic_\n- Lists and structured answers\n"
            "Always consult a healthcare provider for personal advice."
        ),
        "sources": ["Knowledge base (demo)", "General medical literature"],
        "confidence": "medium",
        "suggested_questions": ["What are the main types of diabetes?", "How is blood sugar managed?"],
    }


def render_chat():
    init_state()  # must run first so dark_mode (and messages) exist
    inject_theme(st.session_state.dark_mode)
    render_sidebar()

    st.markdown('<div class="med-strip"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="medibot-hero">
            <h1>🩺 MediBot</h1>
            <div class="tagline">Context-aware medical Q&A — ask from our knowledge base.</div>
            <span class="badge">Medical Q&A • Evidence-based</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scrollable chat container
    chat_container = st.container()
    with chat_container:
        for i, m in enumerate(st.session_state.messages):
            raw = msg_normalize(m)
            role = raw["role"]
            avatar = "🩺" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.markdown(raw["content"])
                if raw["timestamp"]:
                    try:
                        ts = datetime.fromisoformat(raw["timestamp"]).strftime("%b %d, %H:%M")
                    except Exception:
                        ts = raw["timestamp"]
                    st.caption(f"_{ts}_")
                if role == "assistant" and (raw["sources"] or raw["confidence"]):
                    extras = []
                    if raw["confidence"]:
                        c = raw["confidence"].lower()
                        cls = "confidence-high" if c == "high" else "confidence-medium" if c == "medium" else "confidence-low"
                        extras.append(f'<span class="{cls}">Confidence: **{raw["confidence"].title()}**</span>')
                    if raw["sources"]:
                        extras.append("Sources: " + "; ".join(raw["sources"]))
                    if extras:
                        st.markdown(" — ".join(extras), unsafe_allow_html=True)
                if role == "assistant":
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.download_button("📋 Copy / Download", data=raw["content"], file_name="medibot_response.txt", mime="text/plain", key=f"dl_{i}", use_container_width=True)
                    with col2:
                        if i == len(st.session_state.messages) - 1 and not st.session_state.get("is_processing"):
                            if st.button("🔄 Regenerate", key=f"regen_{i}", use_container_width=True):
                                if len(st.session_state.messages) >= 2:
                                    st.session_state.messages.pop()  # remove last assistant reply
                                    st.session_state.is_processing = True
                                st.rerun()
                if role == "assistant" and raw["suggested_questions"]:
                    st.markdown("**Follow-up:**")
                    for j, q in enumerate(raw["suggested_questions"][:3]):
                        if st.button(q[:60] + ("…" if len(q) > 60 else ""), key=f"follow_{i}_{j}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": q, "timestamp": datetime.now().isoformat()})
                            st.session_state.is_processing = True
                            st.rerun()

        if st.session_state.get("is_processing"):
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown("MediBot is thinking…")
                st.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)

    # Not medical advice banner (near input)
    st.markdown(
        '<div class="trust-banner">⚠️ <strong>Not medical advice.</strong> This is for information only. Always consult a healthcare professional.</div>',
        unsafe_allow_html=True,
    )

    # Input: chat_input is simple; we add character hint via placeholder
    prompt = st.chat_input("Ask a medical question (Enter to send)")
    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
        })
        st.session_state.is_processing = True
        st.rerun()

    # Process pending response - USE REAL LLM
    if st.session_state.get("is_processing") and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        time.sleep(1.2)
        last_user = st.session_state.messages[-1]["content"]
        
        # Import and get QA chain
        try:
            from connect_memory_with_llm import create_qa_chain
            chain = create_qa_chain()
            
            if chain is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ Error: Unable to initialize QA chain. Please check:\n1. OPENAI_API_KEY is set in .env file\n2. Vectorstore exists at vectorstore/db_faiss/",
                    "timestamp": datetime.now().isoformat(),
                    "sources": [],
                    "confidence": None,
                    "suggested_questions": [],
                })
            else:
                # Call the real LLM
                with st.spinner("🤔 Thinking..."):
                    res = chain.invoke({"query": last_user})
                sources = [doc.metadata.get("source", "Unknown") for doc in res.get("source_documents", [])]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["result"],
                    "timestamp": datetime.now().isoformat(),
                    "sources": list(set(sources)),
                    "confidence": None,
                    "suggested_questions": [],
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}\n\nPlease check:\n1. OPENAI_API_KEY is valid\n2. Vectorstore exists\n3. See terminal for details",
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "confidence": None,
                "suggested_questions": [],
            })
        
        st.session_state.is_processing = False
        st.rerun()

    # Scroll to bottom link
    st.markdown('[⬇️ Scroll to bottom](#bottom)', unsafe_allow_html=True)
    st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)


def main():
    render_chat()


if __name__ == "__main__":
    main()
