import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "../knowledge_base"
FAISS_PATH = "../faiss_index"

# Category folders
CATEGORY_PATHS = {
    "educational": os.path.join(DATA_PATH, "educational"),
    "legal": os.path.join(DATA_PATH, "legal"),
    "general": DATA_PATH,  # fallback: all documents
}

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


def load_docs(path):
    docs = []

    if os.path.exists(path):
        # TXT files
        try:
            txt_loader = DirectoryLoader(
                path, glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            docs.extend(txt_loader.load())
        except Exception:
            pass

        # PDF files
        try:
            pdf_loader = DirectoryLoader(
                path, glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            docs.extend(pdf_loader.load())
        except Exception:
            pass

    return docs


# Build index for each category
for category, path in CATEGORY_PATHS.items():
    print(f"\n📂 Processing: {category} ({path})")

    docs = load_docs(path)

    if not docs:
        print(f"  ⚠️ No documents found, skipping...")
        continue

    chunks = splitter.split_documents(docs)
    print(f"  📄 {len(docs)} docs → {len(chunks)} chunks")

    db = FAISS.from_documents(chunks, embeddings)

    index_path = os.path.join(FAISS_PATH, category)
    db.save_local(index_path)
    print(f"  ✅ Index saved to {index_path}")


# Also build a combined index
print(f"\n📂 Processing: combined (all documents)")
all_docs = load_docs(DATA_PATH)

for category, path in CATEGORY_PATHS.items():
    if path != DATA_PATH:
        all_docs.extend(load_docs(path))

if all_docs:
    chunks = splitter.split_documents(all_docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(os.path.join(FAISS_PATH, "combined"))
    print(f"  ✅ Combined index saved")

print("\n🎉 All indexes created!")