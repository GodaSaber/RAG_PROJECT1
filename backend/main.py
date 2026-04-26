import os
import json
import torch
import tempfile
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from classifier import QuestionClassifier, SimpleTokenizer, CATEGORIES


# =========================
# Load env
# =========================
load_dotenv()

FAISS_PATH = "../faiss_index"
MODEL_DIR = "../models"


# =========================
# Load Classifier
# =========================
print("🧠 Loading classifier...")

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

tokenizer = SimpleTokenizer()
tokenizer.load(os.path.join(MODEL_DIR, "tokenizer.json"))

classifier = QuestionClassifier(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_classes=config["num_classes"],
    max_len=config["max_len"],
)
classifier.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "classifier.pth"), map_location="cpu")
)
classifier.eval()

print("✅ Classifier loaded!")


def classify_text(text: str):

    # Neural network)
    with torch.no_grad():
        encoded = torch.tensor(
            [tokenizer.encode(text, config["max_len"])], dtype=torch.long
        )
        output = classifier(encoded)
        probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, 1).item()
        confidence = probs[0][predicted].item()

    print(f"  🧠 Neural Network → {CATEGORIES[predicted]} ({confidence*100:.1f}%)")

    return CATEGORIES[predicted], confidence


def classify_document(docs):
    """Classify a document by analyzing its content"""
    # Take sample from multiple chunks
    sample_text = " ".join([doc.page_content[:300] for doc in docs[:10]])
    category, confidence = classify_text(sample_text)
    print(f"  📋 Document classified as: {category} ({confidence*100:.1f}%)")
    return category


# =========================
# Embeddings
# =========================
print("📊 Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


# =========================
# FAISS Index Manager
# =========================
class IndexManager:
    def __init__(self):
        self.indexes = {}
        self.load_all_indexes()

    def load_all_indexes(self):
        """Load all existing FAISS indexes"""
        self.indexes = {}
        categories = ["educational", "legal", "general", "combined"]

        for category in categories:
            index_path = os.path.join(FAISS_PATH, category)
            if os.path.exists(index_path):
                try:
                    db = FAISS.load_local(
                        index_path, embeddings, allow_dangerous_deserialization=True
                    )
                    self.indexes[category] = db
                    print(f"  ✅ Loaded index: {category}")
                except Exception as e:
                    print(f"  ❌ Failed to load {category}: {e}")

        print(f"📚 {len(self.indexes)} indexes loaded")

    def add_documents(self, docs, category):
        """Add documents to the correct category index"""
        chunks = splitter.split_documents(docs)

        if category in self.indexes:
            self.indexes[category].add_documents(chunks)
        else:
            self.indexes[category] = FAISS.from_documents(chunks, embeddings)

        # Save
        index_path = os.path.join(FAISS_PATH, category)
        os.makedirs(index_path, exist_ok=True)
        self.indexes[category].save_local(index_path)

        print(f"  ✅ Added {len(chunks)} chunks to '{category}'")
        return len(chunks)

    def get_retriever(self, category, k=2):
        """Get retriever for a specific category"""
        if category in self.indexes:
            return self.indexes[category].as_retriever(search_kwargs={"k": k})
        return None

    def search_all(self, query, k=3):
        """Search across ALL indexes and return best results"""
        all_docs = []

        for category, db in self.indexes.items():
            try:
                docs_with_scores = db.similarity_search_with_score(query, k=k)
                for doc, score in docs_with_scores:
                    doc.metadata["_category"] = category
                    doc.metadata["_score"] = float(score)
                    all_docs.append((doc, score))
            except Exception as e:
                print(f"  ⚠️ Error searching {category}: {e}")

        # Sort by score (lower = better in FAISS)
        all_docs.sort(key=lambda x: x[1])
        return [doc for doc, score in all_docs[:k]]

    def get_index_info(self):
        """Get info about all indexes"""
        info = {}
        for category, db in self.indexes.items():
            try:
                count = db.index.ntotal
            except Exception:
                count = "unknown"
            info[category] = {"chunks": count}
        return info


# Initialize
index_manager = IndexManager()


# =========================
# LLM
# =========================
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.5,
    num_predict=200,
    keep_alive=-1,
)


# =========================
# Prompt with Chat History
# =========================
SYSTEM_PROMPT = """
You are a helpful assistant.
Use ONLY the context below to answer in max 3 sentences.
If the context doesn't contain the answer, say you don't know.
Answer in the same language as the question.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# =========================
# Chat History Storage
# =========================
chat_histories = {}


# =========================
# Helper: Smart search
# =========================
def smart_search(question: str):
    """Classify question and search in the right index"""
    category, confidence = classify_text(question)
    print(f"🏷️ Question classified as: {category} ({confidence*100:.1f}%)")

    docs = []
    actual_category = category

    # Step 1: If high confidence → search ONLY in that category
    if confidence > 0.6 and category in index_manager.indexes:
        retriever = index_manager.get_retriever(category, k=3)
        docs = retriever.invoke(question)
        print(f"🔍 Searching in: {category} only")

        # Check if docs are relevant (score check)
        if docs:
            # Tag docs with category
            for doc in docs:
                doc.metadata["_category"] = category
            actual_category = category
        else:
            # No docs found → fallback to search_all
            print(f"  ⚠️ No docs in {category}, falling back to all")
            docs = index_manager.search_all(question, k=3)

    # Step 2: Low confidence → search ALL but prefer classified category
    else:
        print(f"🔍 Low confidence, searching all indexes")

        # Search in classified category first
        category_docs = []
        if category in index_manager.indexes:
            try:
                retriever = index_manager.get_retriever(category, k=2)
                category_docs = retriever.invoke(question)
                for doc in category_docs:
                    doc.metadata["_category"] = category
            except Exception:
                pass

        # Search in all other indexes
        other_docs = []
        for cat_name, db in index_manager.indexes.items():
            if cat_name == category:
                continue
            try:
                results = db.similarity_search_with_score(question, k=2)
                for doc, score in results:
                    doc.metadata["_category"] = cat_name
                    doc.metadata["_score"] = float(score)
                    other_docs.append((doc, score))
            except Exception:
                pass

        # Sort other docs by score
        other_docs.sort(key=lambda x: x[1])
        other_docs = [doc for doc, score in other_docs[:2]]

        # Combine: category docs first, then others
        docs = category_docs + other_docs
        docs = docs[:3]  # Max 3 docs

    # Step 3: Determine actual category from docs
    if docs:
        doc_categories = [doc.metadata.get("_category", category) for doc in docs]
        actual_category = Counter(doc_categories).most_common(1)[0][0]

    return docs, actual_category, confidence


# =========================
# Helper: extract sources
# =========================
def extract_sources(docs):
    """Extract source info from retrieved documents"""
    sources = []
    seen_files = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(source)
        page = doc.metadata.get("page", "")
        doc_category = doc.metadata.get("_category", "")

        # Avoid duplicate files in sources
        file_key = f"{filename}_p{page}"
        if file_key in seen_files:
            continue
        seen_files.add(file_key)

        sources.append(
            {
                "file": filename,
                "page": page,
                "category": doc_category,
                "preview": doc.page_content[:150] + "...",
            }
        )

    return sources


# =========================
# FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Models
# =========================
class Query(BaseModel):
    text: str
    session_id: str = "default"


class Feedback(BaseModel):
    question: str
    answer: str
    rating: str


# =========================
# Endpoints
# =========================


@app.get("/")
def home():
    return {"message": "RAG API with Smart Classification running"}


# ---------- 1. Main Query ----------
@app.post("/query")
def query_rag(query: Query):
    print(f"\n{'='*50}")
    print(f"❓ Question: {query.text}")

    # Step 1: Smart search
    docs, category, confidence = smart_search(query.text)

    # Step 2: Get chat history
    if query.session_id not in chat_histories:
        chat_histories[query.session_id] = []
    history = chat_histories[query.session_id]

    # Step 3: Generate answer
    context = "\n".join([doc.page_content for doc in docs])

    messages = prompt.format_messages(
        input=query.text, context=context, chat_history=history
    )
    response = llm.invoke(messages)
    answer = response.content

    # Step 4: Save history
    history.append(HumanMessage(content=query.text))
    history.append(AIMessage(content=answer))
    chat_histories[query.session_id] = history[-10:]

    # Step 5: Sources
    sources = extract_sources(docs)

    print(f"✅ Category: {category} | Sources: {[s['file'] for s in sources]}")
    print(f"{'='*50}")

    return {
        "answer": answer,
        "category": category,
        "confidence": round(confidence * 100, 1),
        "sources": sources,
    }


# ---------- 2. Streaming Query ----------
@app.post("/query/stream")
async def query_rag_stream(query: Query):
    def generate():
        docs, category, confidence = smart_search(query.text)

        if query.session_id not in chat_histories:
            chat_histories[query.session_id] = []
        history = chat_histories[query.session_id]

        context = "\n".join([doc.page_content for doc in docs])

        messages = prompt.format_messages(
            input=query.text, context=context, chat_history=history
        )

        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        history.append(HumanMessage(content=query.text))
        history.append(AIMessage(content=full_response))
        chat_histories[query.session_id] = history[-10:]

        sources = extract_sources(docs)
        metadata = json.dumps(
            {
                "category": category,
                "confidence": round(confidence * 100, 1),
                "sources": sources,
            },
            ensure_ascii=False,
        )
        yield f"\n<!--METADATA:{metadata}:METADATA-->"

    return StreamingResponse(generate(), media_type="text/plain")


# ---------- 3. File Upload (Auto-classify) ----------
@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload files — auto-classify and add to correct index"""
    results = []

    for file in files:
        print(f"\n📤 Uploading: {file.filename}")

        suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load document
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()

            if not docs:
                results.append(
                    {
                        "file": file.filename,
                        "status": "failed",
                        "error": "No content found",
                    }
                )
                continue

            # Auto-classify the document content
            category = classify_document(docs)

            # Add to the correct index
            num_chunks = index_manager.add_documents(docs, category)

            results.append(
                {
                    "file": file.filename,
                    "category": category,
                    "chunks": num_chunks,
                    "status": "success",
                }
            )

            print(f"  ✅ {file.filename} → {category} ({num_chunks} chunks)")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"file": file.filename, "status": "failed", "error": str(e)})

        finally:
            os.unlink(tmp_path)

    return {"results": results}


# ---------- 4. Classify Only ----------
@app.post("/classify")
def classify_only(query: Query):
    category, confidence = classify_text(query.text)
    return {
        "question": query.text,
        "category": category,
        "confidence": round(confidence * 100, 1),
    }


# ---------- 5. Feedback ----------
@app.post("/feedback")
def save_feedback(feedback: Feedback):
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": feedback.question,
        "answer": feedback.answer,
        "rating": feedback.rating,
    }

    with open("feedback.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"📝 Feedback: {feedback.rating}")
    return {"message": "Feedback saved"}


# ---------- 6. Clear History ----------
@app.post("/clear-history")
def clear_history(session_id: str = "default"):
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": f"History cleared for: {session_id}"}


# ---------- 7. Health Check ----------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "indexes": index_manager.get_index_info(),
        "active_sessions": len(chat_histories),
    }


# ---------- 8. List Indexes ----------
@app.get("/indexes")
def list_indexes():
    return index_manager.get_index_info()


# ---------- 9. Sessions ----------
@app.get("/sessions")
def get_sessions():
    sessions = []
    for sid, history in chat_histories.items():
        if history:
            title = "New Chat"
            for msg in history:
                if isinstance(msg, HumanMessage):
                    title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                    break
            sessions.append(
                {
                    "session_id": sid,
                    "title": title,
                    "messages_count": len(history),
                }
            )
    sessions.reverse()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
def get_session_history(session_id: str):
    if session_id not in chat_histories:
        return {"messages": []}

    messages = []
    for msg in chat_histories[session_id]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "ai", "content": msg.content})

    return {"messages": messages}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
        return {"message": f"Session {session_id} deleted"}
    return {"message": "Session not found"}
