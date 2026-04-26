# 🤖 RAG Document Chatbot with Neural Network Classifier

An intelligent chatbot that answers your questions by searching through your own documents. It uses a custom-built Neural Network to classify questions into categories (Educational, Legal, General), then searches the right document index for accurate answers.

Built with RAG (Retrieval-Augmented Generation) technology combining vector search, neural network classification, and Large Language Models.

Supports **Arabic** and **English** with both local (Ollama) and cloud (Gemini) LLM options.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Network-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)

---

## 📖 Table of Contents

- [About The Project](#-about-the-project)
- [System Architecture](#-system-architecture)
- [Neural Network Classifier](#-neural-network-classifier)
- [Project Structure](#-project-structure)
- [How To Run](#-how-to-run)

---

## 📖 About The Project

### What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant information from your documents, then using that context to generate accurate answers. Instead of relying only on the LLM's training data, it grounds answers in your actual documents.

### What Makes This Project Special?

This is not a basic RAG project. It includes a **custom Neural Network** built from scratch using PyTorch that classifies every question before searching. This means the system searches in the right category of documents, giving more accurate answers and correct source attribution.

### How It Works

```
User Question: "What is lexical analysis?"
│
▼
┌─────────────────────────────┐
│ 1. Keyword Detection        │
│    "lexical" found          │
│    → Educational            │
│                             │
│ 2. Neural Network (backup)  │
│    CNN Classifier           │
│    → Confirms Educational   │
└─────────────────────────────┘
│
▼ Category: Educational (95%)
┌─────────────────────────────┐
│ 3. FAISS Vector Search      │
│    Search ONLY in           │
│    educational/ index       │
│    → Finds relevant chunks  │
└─────────────────────────────┘
│
▼ Retrieved: compiler_basics.pdf
┌─────────────────────────────┐
│ 4. LLM (Qwen 2.5)           │
│    Generate answer using    │
│    retrieved context        │
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ Answer + Category + Sources │
│ 📚 Educational (95%)        │
│ 📄 compiler_basics.pdf p.3  │
└─────────────────────────────┘
```

### Complete Pipeline

1. **📄 Document Loading** — Reads PDF and TXT files
2. **🧠 Document Classification** — Neural Network classifies document content
3. **✂️ Text Splitting** — Breaks documents into 1000-char chunks with 150 overlap
4. **🔢 Embedding** — Converts chunks to vectors using HuggingFace multilingual model
5. **💾 Categorized Indexing** — Stores vectors in separate FAISS indexes per category
6. **❓ Question Classification** — Keywords + Neural Network classify the question
7. **🔍 Smart Retrieval** — Searches ONLY in the matching category index
8. **🤖 Generation** — LLM generates answer using retrieved context + chat history
9. **📚 Source Attribution** — Shows which file and page the answer came from

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                     STREAMLIT UI                     │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │  Chat   │  │  Upload  │  │  Session Management │ │
│  │ History │  │  Files   │  │  Streaming Toggle   │ │
│  └─────────┘  └──────────┘  └─────────────────────┘ │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP / Streaming
┌──────────────────────▼───────────────────────────────┐
│                   FASTAPI BACKEND                    │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │            Question Classifier               │   │
│  │  ┌─────────────┐   ┌──────────────────────┐  │   │
│  │  │  Keywords   │──▶│   Neural Network     │  │   │
│  │  │  Detection  │   │   (CNN - PyTorch)    │  │   │
│  │  └─────────────┘   └──────────────────────┘  │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │ Category                   │
│  ┌──────────────────────▼───────────────────────┐   │
│  │               Index Manager                  │   │
│  │  ┌────────────┐  ┌───────┐  ┌─────────┐     │   │
│  │  │Educational │  │Legal  │  │General  │     │   │
│  │  │FAISS Index │  │FAISS  │  │FAISS    │     │   │
│  │  └────────────┘  └───────┘  └─────────┘     │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │ Retrieved Docs             │
│  ┌──────────────────────▼───────────────────────┐   │
│  │           LLM (Ollama - Qwen 2.5)            │   │
│  │           + Chat History                     │   │
│  │           + Streaming Support                │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 🧠 Neural Network Classifier

### Architecture

The classifier is a **1D Convolutional Neural Network (CNN)** built from scratch:

```
Input: Question Text
│
▼
┌─────────────────────┐
│   Simple Tokenizer  │  Splits Arabic + English text
│   (Custom Built)    │  Converts words → indices
└─────────┬───────────┘
          │
          ▼ Shape: (batch, 50)
┌─────────────────────┐
│   Embedding Layer   │  vocab_size → 64 dims
└─────────┬───────────┘
          │
          ▼ Shape: (batch, 64, 50)
┌─────────────────────┐
│   Conv1D Layer 1    │  64 → 128 channels, kernel=3
│   + ReLU + Dropout  │
└─────────┬───────────┘
          │
          ▼ Shape: (batch, 128, 50)
┌─────────────────────┐
│   Conv1D Layer 2    │  128 → 128 channels, kernel=3
│   + ReLU            │
└─────────┬───────────┘
          │
          ▼ Shape: (batch, 128, 50)
┌─────────────────────┐
│  AdaptiveMaxPool1D  │  → (batch, 128, 1)
└─────────┬───────────┘
          │
          ▼ Shape: (batch, 128)
┌─────────────────────┐
│   FC Layer 1        │  128 → 64 + ReLU + Dropout
│   FC Layer 2        │  64 → 3 (num_classes)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Output: 3 classes  │
│  0: Educational     │
│  1: Legal           │
│  2: General         │
└─────────────────────┘
```

### Custom Tokenizer

The tokenizer is built from scratch to handle both Arabic and English:

```python
# Splits text into Arabic and English words using regex
# Arabic: [\u0600-\u06FF]+
# English: [a-zA-Z]+
# Numbers: [0-9]+

Input:  "ما هو lexical analysis في compilers"
Output: ["ما", "هو", "lexical", "analysis", "في", "compilers"]
```

### Classification Strategy

The system uses a two-layer classification approach:

```
Question
    │
    ▼
┌──────────────────┐
│ Layer 1: Keywords │  ← Fast, accurate for known terms
│  edu_score: 3    │
│  legal_score: 0  │
│  → Educational 95%│
└──────────────────┘
    │
    │ If no keywords match:
    ▼
┌──────────────────┐
│  Layer 2: CNN    │  ← Handles unknown/ambiguous questions
│  Neural Network  │
│  → Prediction    │
└──────────────────┘
```

### Training Data

- 90+ samples per category (Educational, Legal, General)
- Bilingual: Arabic + English examples
- Data augmentation through varied phrasing

### Training Results

```
Epoch [100/100]  Loss: 0.0123  Accuracy: 100.0%

Testing:
  'what is inheritance in java'     → educational (98.5%)
  'ما هي المادة الأولى من القانون'  → legal (97.2%)
  'hello how are you'               → general (99.1%)
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── knowledge_base/              # 📄 Your documents organized by category
│   ├── educational/             # Programming, CS, AI documents
│   │   ├── oops_java.pdf
│   │   ├── webscraping.txt
│   │   └── compilers.pdf
│   └── legal/                   # Law documents
│       └── criminal_law.pdf
│
├── faiss_index/                 # 💾 Auto-generated FAISS indexes
│   ├── educational/
│   ├── legal/
│   └── general/
│
├── models/                      # 🧠 Trained neural network
│   ├── classifier.pth           # Model weights
│   ├── tokenizer.json           # Vocabulary
│   └── config.json              # Model configuration
│
├── backend/
│   ├── classifier.py            # 🧠 Neural Network + Tokenizer classes
│   ├── train_classifier.py      # 🏋️ Training script
│   ├── index.py                 # 🔢 Build FAISS indexes
│   ├── main.py                  # ⚡ FastAPI server + RAG chain
│   └── app.py                   # 💬 Streamlit chat interface
│
├── .env.example                 # 🔑 Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 How To Run

### Prerequisites

- Python 3.10
- Conda (Miniconda or Anaconda)
- Git
- ~4GB free disk space

---

### Step 1: Clone The Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2: Create Conda Environment

```bash
conda create -n RAG python=3.10 -y
conda activate RAG
```

### Step 3: Install Dependencies

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Setup LLM

#### 🖥️ Option A: Ollama — Local (Recommended)

> No API key needed. No internet needed. No rate limits.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

#### ☁️ Option B: Google Gemini API — Cloud

> Faster but requires an API key and has rate limits.

```bash
cp .env.example .env
```

Get an API key from [Google AI Studio](https://aistudio.google.com/) and add it to `.env`:

```text
GOOGLE_API_KEY=your_api_key_here
```

Update `main.py` to use Gemini:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

### Step 5: Add Your Documents

```bash
mkdir -p knowledge_base/educational
mkdir -p knowledge_base/legal
```

Place files in the appropriate folders:

```
knowledge_base/
├── educational/         # Programming, CS, AI, Math docs
│   ├── your_file.pdf
│   └── your_file.txt
└── legal/               # Law, regulations docs
    └── your_file.pdf
```

### Step 6: Train The Neural Network Classifier

```bash
cd backend
python train_classifier.py
```

Expected output:

```
==================================================
🧠 Training Question Classifier
==================================================
📝 Vocab size: 350
📊 Training samples: 150
Epoch [10/100]   Loss: 0.8234  Accuracy: 72.2%
Epoch [50/100]   Loss: 0.0567  Accuracy: 97.8%
Epoch [100/100]  Loss: 0.0123  Accuracy: 100.0%
==================================================
✅ Model saved to ../models/
==================================================
```

### Step 7: Build The FAISS Indexes

```bash
python index.py
```

Expected output:

```
📂 Processing: educational
  📄 5 docs → 60 chunks
  ✅ Index saved
📂 Processing: legal
  📄 1 docs → 45 chunks
  ✅ Index saved
🎉 All indexes created!
```

### Step 8: Run The Application

Open **3 separate terminals** and run the following:

**Terminal 1 — Start Ollama** *(skip if using Gemini)*

```bash
ollama serve
```

**Terminal 2 — Start Backend API**

```bash
conda activate RAG
cd backend
uvicorn main:app --reload
```

Expected output:

```
🧠 Loading classifier...
✅ Classifier loaded!
📊 Loading embeddings...
💾 Loading FAISS indexes...
  ✅ Loaded index: educational
  ✅ Loaded index: legal
📚 2 indexes loaded!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 3 — Start Frontend**

```bash
conda activate RAG
cd backend
streamlit run app.py
```

### Step 9: Open The App

Go to 👉 **http://localhost:8501**

Start chatting with your documents! 🎉