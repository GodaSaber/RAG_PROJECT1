import streamlit as st
import requests
import uuid
import json

# =========================
# Config
# =========================
API_BASE = "http://127.0.0.1:8000"
API_QUERY = f"{API_BASE}/query"
API_STREAM = f"{API_BASE}/query/stream"
API_UPLOAD = f"{API_BASE}/upload"
API_FEEDBACK = f"{API_BASE}/feedback"
API_CLEAR = f"{API_BASE}/clear-history"
API_INDEXES = f"{API_BASE}/indexes"

st.set_page_config(page_title="Document Chatbot", page_icon="🤖", layout="centered")

# =========================
# Session State Init
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}

# =========================
# Custom CSS
# =========================
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #4A90D9;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-top: 8px;
    }
    .cat-educational {
        background: #1B5E20;
        color: #A5D6A7;
    }
    .cat-legal {
        background: #B71C1C;
        color: #EF9A9A;
    }
    .cat-general {
        background: #1565C0;
        color: #90CAF9;
    }
    .index-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 10px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .index-educational { background: #1B5E20; color: #A5D6A7; }
    .index-legal { background: #B71C1C; color: #EF9A9A; }
    .index-general { background: #1565C0; color: #90CAF9; }
    .index-combined { background: #4A148C; color: #CE93D8; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Helper Functions
# =========================
CATEGORY_CONFIG = {
    "educational": {"emoji": "📚", "label": "Educational", "css": "cat-educational"},
    "legal": {"emoji": "⚖️", "label": "Legal", "css": "cat-legal"},
    "general": {"emoji": "💬", "label": "General", "css": "cat-general"},
}


def show_category_badge(category, confidence):
    """Display a styled category badge"""
    cfg = CATEGORY_CONFIG.get(
        category, {"emoji": "❓", "label": category, "css": "cat-general"}
    )
    st.markdown(
        f"""<span class='category-badge {cfg["css"]}'>
            {cfg["emoji"]} {cfg["label"]} — 🎯 {confidence}%
        </span>""",
        unsafe_allow_html=True,
    )


def show_sources(sources):
    """Display sources with category info"""
    if not sources:
        return

    with st.expander("📚 Sources"):
        for j, src in enumerate(sources, 1):
            page_info = f" — Page {src['page']}" if src.get("page") else ""
            cat = src.get("category", "")
            cat_emoji = CATEGORY_CONFIG.get(cat, {}).get("emoji", "📄")

            st.markdown(
                f"**{j}. {cat_emoji} {src['file']}{page_info}**"
            )
            if cat:
                st.caption(f"Category: {cat}")
            st.caption(src.get("preview", ""))
            st.markdown("---")


def parse_stream_response(full_text):
    """Separate answer text from metadata in stream response"""
    display_text = full_text.split("<!--METADATA:")[0].strip()
    category = ""
    confidence = 0
    sources = []

    if "<!--METADATA:" in full_text and ":METADATA-->" in full_text:
        try:
            meta_str = full_text.split("<!--METADATA:")[1].split(":METADATA-->")[0]
            meta = json.loads(meta_str)
            category = meta.get("category", "")
            confidence = meta.get("confidence", 0)
            sources = meta.get("sources", [])
        except Exception as e:
            print(f"Metadata parse error: {e}")

    return display_text, category, confidence, sources


def save_current_session():
    """Save current session to all_sessions"""
    if st.session_state.messages:
        title = "New Chat"
        for msg in st.session_state.messages:
            if msg["role"] == "human":
                title = msg["content"][:50]
                if len(msg["content"]) > 50:
                    title += "..."
                break

        st.session_state.all_sessions[st.session_state.session_id] = {
            "title": title,
            "messages": list(st.session_state.messages),
        }


def load_session(session_id):
    """Load a saved session"""
    if session_id in st.session_state.all_sessions:
        session_data = st.session_state.all_sessions[session_id]
        st.session_state.session_id = session_id
        st.session_state.messages = list(session_data["messages"])


def create_new_session():
    """Save current session and create a new one"""
    save_current_session()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def get_loaded_indexes():
    """Get index info from backend"""
    try:
        res = requests.get(API_INDEXES, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {}


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chatbot.png", width=80)
    st.markdown("## 📚 Document Chatbot")
    st.markdown("---")

    # ===== New Chat Button =====
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.markdown("---")

    # ===== Chat History =====
    st.markdown("### 💬 Chat History")
    save_current_session()

    if st.session_state.all_sessions:
        session_ids = list(st.session_state.all_sessions.keys())
        if st.session_state.session_id in session_ids:
            session_ids.remove(st.session_state.session_id)
            session_ids.insert(0, st.session_state.session_id)

        for sid in session_ids:
            session_data = st.session_state.all_sessions[sid]
            title = session_data.get("title", "New Chat")
            msg_count = len(session_data.get("messages", []))
            is_current = sid == st.session_state.session_id
            icon = "💬" if is_current else "🗨️"

            col_chat, col_del = st.columns([5, 1])

            with col_chat:
                if st.button(
                    f"{icon} {title}",
                    key=f"session_{sid}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    disabled=is_current,
                ):
                    save_current_session()
                    load_session(sid)
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{sid}", help="Delete this chat"):
                    if sid in st.session_state.all_sessions:
                        del st.session_state.all_sessions[sid]
                    if sid == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    try:
                        requests.post(API_CLEAR, params={"session_id": sid}, timeout=5)
                    except Exception:
                        pass
                    st.rerun()

            if not is_current and msg_count > 0:
                st.caption(f"    📝 {msg_count // 2} messages")
    else:
        st.caption("No chat history yet")

    st.markdown("---")

    # ===== Streaming Toggle =====
    st.session_state.use_streaming = st.toggle(
        "⚡ Streaming Mode", value=st.session_state.use_streaming
    )

    st.markdown("---")

    # ===== Upload Section (Auto-classify) =====
    st.markdown("### 📤 Upload Documents")
    st.caption("Files are auto-classified by content 🧠")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button("📥 Process & Classify", use_container_width=True):
        with st.spinner("🧠 Classifying & indexing..."):
            files = []
            for f in uploaded_files:
                files.append(("files", (f.name, f.read(), f.type)))

            try:
                res = requests.post(API_UPLOAD, files=files, timeout=120)

                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])

                    for result in results:
                        if result["status"] == "success":
                            cat_cfg = CATEGORY_CONFIG.get(
                                result["category"],
                                {"emoji": "📄", "label": result["category"]}
                            )
                            st.success(
                                f"✅ {result['file']}\n"
                                f"{cat_cfg['emoji']} Auto-classified: "
                                f"**{result['category']}** — "
                                f"{result['chunks']} chunks"
                            )
                        else:
                            st.error(
                                f"❌ {result['file']}: {result.get('error', 'Unknown error')}"
                            )
                else:
                    st.error(f"❌ Upload failed: {res.text}")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

    st.markdown("---")

    # ===== Loaded Indexes (Live from backend) =====
    st.markdown("### 📊 Loaded Indexes")

    indexes = get_loaded_indexes()
    if indexes:
        for cat_name, info in indexes.items():
            css_class = f"index-{cat_name}"
            cat_cfg = CATEGORY_CONFIG.get(
                cat_name, {"emoji": "📦", "label": cat_name}
            )
            chunks = info.get("chunks", "?")
            st.markdown(
                f"<span class='index-badge {css_class}'>"
                f"{cat_cfg.get('emoji', '📦')} {cat_name}: {chunks} chunks"
                f"</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("⚠️ Cannot connect to API")

    st.markdown("---")

    # ===== Example Questions =====
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is web scraping?",
        "Explain OOP in Java",
        "What are the 4 pillars of OOP?",
        "How does BeautifulSoup work?",
        "ما هي عقوبة السرقة؟",
        "ما هي المادة الأولى من القانون؟",
    ]
    for q in example_questions:
        if st.button(f"➜ {q}", key=q, use_container_width=True):
            st.session_state["example_question"] = q

    st.markdown("---")

    # ===== Actions =====
    st.markdown("### ⚙️ Actions")

    if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
        st.session_state.all_sessions = {}
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.caption(f"Total chats: {len(st.session_state.all_sessions)}")
    st.caption("Powered by Qwen + LangChain + FAISS")


# =========================
# Header
# =========================
st.markdown(
    """
<div class='main-header'>
    <h1>🤖 Document Chatbot</h1>
    <p>Ask anything about your uploaded documents</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# Welcome message
# =========================
if len(st.session_state.messages) == 0:
    st.markdown(
        """
    <div style='text-align:center; padding: 2rem; color: #888;'>
        <p style='font-size: 3rem;'>💬</p>
        <p style='font-size: 1.2rem;'>Start a conversation by typing below<br>
        or pick an example from the sidebar!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =========================
# Display chat history
# =========================
for i, message in enumerate(st.session_state.messages):
    avatar = "🧑" if message["role"] == "human" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "ai":
            # Show category badge
            if message.get("category"):
                show_category_badge(
                    message["category"],
                    message.get("confidence", 0),
                )

            # Show sources
            show_sources(message.get("sources", []))

            # Feedback buttons
            fcol1, fcol2, fcol3 = st.columns([1, 1, 8])
            with fcol1:
                if st.button("👍", key=f"up_{i}"):
                    try:
                        human_msg = ""
                        if i > 0:
                            human_msg = st.session_state.messages[i - 1].get(
                                "content", ""
                            )
                        requests.post(
                            API_FEEDBACK,
                            json={
                                "question": human_msg,
                                "answer": message["content"],
                                "rating": "positive",
                            },
                            timeout=10,
                        )
                        st.toast("Thanks! 👍")
                    except Exception:
                        st.toast("Could not save feedback")
            with fcol2:
                if st.button("👎", key=f"down_{i}"):
                    try:
                        human_msg = ""
                        if i > 0:
                            human_msg = st.session_state.messages[i - 1].get(
                                "content", ""
                            )
                        requests.post(
                            API_FEEDBACK,
                            json={
                                "question": human_msg,
                                "answer": message["content"],
                                "rating": "negative",
                            },
                            timeout=10,
                        )
                        st.toast("Thanks for feedback! 👎")
                    except Exception:
                        st.toast("Could not save feedback")

# =========================
# Handle example question
# =========================
if "example_question" in st.session_state:
    prompt = st.session_state.pop("example_question")
else:
    prompt = st.chat_input("💬 Ask me anything about your documents...")

# =========================
# Process prompt
# =========================
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human", avatar="🧑"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("ai", avatar="🤖"):

        # ===== STREAMING MODE =====
        if st.session_state.use_streaming:
            with st.spinner("🔍 Searching documents & thinking..."):
                try:
                    res = requests.post(
                        API_STREAM,
                        json={
                            "text": prompt,
                            "session_id": st.session_state.session_id,
                        },
                        stream=True,
                        timeout=120,
                    )

                    if res.status_code == 200:
                        full_text = ""
                        placeholder = st.empty()

                        for chunk in res.iter_content(chunk_size=None):
                            if chunk:
                                decoded = chunk.decode("utf-8")
                                full_text += decoded
                                display_text = full_text.split("<!--METADATA:")[0]
                                placeholder.markdown(display_text + "▌")

                        display_text, category, confidence, sources = (
                            parse_stream_response(full_text)
                        )

                        placeholder.markdown(display_text)

                        if category:
                            show_category_badge(category, confidence)

                        show_sources(sources)

                        st.session_state.messages.append(
                            {
                                "role": "ai",
                                "content": display_text,
                                "category": category,
                                "confidence": confidence,
                                "sources": sources,
                            }
                        )

                        save_current_session()

                    else:
                        st.error(f"Error: {res.status_code} - {res.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to backend. "
                        "Make sure the API server is running!"
                    )
                    st.code(
                        "cd ~/rag_project/backend\nuvicorn main:app --reload",
                        language="bash",
                    )
                except requests.exceptions.Timeout:
                    st.error("⏳ Request timed out. Try again.")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

        # ===== NORMAL MODE =====
        else:
            with st.spinner("🔍 Searching documents & thinking..."):
                try:
                    res = requests.post(
                        API_QUERY,
                        json={
                            "text": prompt,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=120,
                    )
                    res.raise_for_status()
                    data = res.json()

                    answer = data["answer"]
                    category = data.get("category", "")
                    confidence = data.get("confidence", 0)
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if category:
                        show_category_badge(category, confidence)

                    show_sources(sources)

                    st.session_state.messages.append(
                        {
                            "role": "ai",
                            "content": answer,
                            "category": category,
                            "confidence": confidence,
                            "sources": sources,
                        }
                    )

                    save_current_session()

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to backend. "
                        "Make sure the API server is running!"
                    )
                    st.code(
                        "cd ~/rag_project/backend\nuvicorn main:app --reload",
                        language="bash",
                    )
                except requests.exceptions.Timeout:
                    st.error("⏳ Request timed out. Try again.")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    st.rerun()