import os
from pathlib import Path
from dotenv import load_dotenv
import time

from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_from_directory
)
from json_logger import write_json_log

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ------------- Config -------------
DATA_DIR = Path("papers")
INDEX_DIR = Path("index/faiss")
ALLOWED_EXTS = {".pdf"}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
RETRIEVAL_K = 8         # final docs
RETRIEVAL_FETCH_K = 48  # candidates for MMR

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

def build_vectorstore(embedding) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(
            f"No PDFs found in {DATA_DIR.resolve()}. Place your files in that folder or upload from the UI."
        )

    all_chunks = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = pdf_path.name
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vs = FAISS.from_documents(all_chunks, embedding)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    return vs

def load_or_create_vectorstore(embedding) -> FAISS:
    if INDEX_DIR.exists():
        try:
            return FAISS.load_local(
                str(INDEX_DIR), embedding, allow_dangerous_deserialization=True
            )
        except Exception:
            pass
    return build_vectorstore(embedding)

def create_app():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env or environment.")

    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    # ---- Models ----
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-3-small"
    )
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0
    )

    # ---- Vectorstore & retriever ----
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K}
    )

    # Ensure sources are returned
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    @app.route("/", methods=["GET", "POST"])
    def index():
        answer, sources, question = "", [], ""
        if request.method == "POST":
            question = (request.form.get("question") or "").strip()
            if question:
                t0 = time.perf_counter()
                result = qa_chain({"query": question})
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                answer = result["result"]
                src_docs = result.get("source_documents", []) or []

                for d in src_docs:
                    snippet = (d.page_content or "")[:400].strip().replace("\n", " ")
                    sources.append({
                        "filename": (d.metadata or {}).get("source", "unknown.pdf"),
                        "snippet": snippet + ("..." if len(d.page_content) > 400 else "")
                    })

                usage = result.get("generation_info") or {}
                prompt_toks = usage.get("prompt_tokens")
                completion_toks = usage.get("completion_tokens")
                total_toks = usage.get("total_tokens")

                retriever_cfg = (
                    f"type={getattr(retriever, 'search_type', 'similarity')};"
                    f"k={getattr(retriever, 'k', None)};"
                    f"fetch_k={getattr(retriever, 'search_kwargs', {}).get('fetch_k')}"
                )

                write_json_log(
                    question=question,
                    answer=answer,
                    latency_ms=elapsed_ms,
                    model=getattr(llm, "model_name", getattr(llm, "model", None)),
                    embedding_model=getattr(embeddings, "model", None),  # <-- fixed name
                    retriever_cfg=retriever_cfg,
                    prompt_tokens=prompt_toks,
                    completion_tokens=completion_toks,
                    total_tokens=total_toks,
                    client_ip=request.headers.get("X-Forwarded-For", request.remote_addr),
                    user_agent=request.headers.get("User-Agent"),
                    source_documents=src_docs,
                )

        return render_template("index.html", answer=answer, sources=sources, question=question)

    @app.route("/upload", methods=["POST"])
    def upload():
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only PDF files are allowed.", "danger")
            return redirect(url_for("index"))

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_path = DATA_DIR / Path(file.filename).name
        file.save(save_path)
        flash(f"Uploaded {save_path.name}. Click 'Rebuild Index' to include it.", "success")
        return redirect(url_for("index"))

    @app.route("/reindex", methods=["POST"])
    def reindex():
        nonlocal vectorstore, retriever
        vectorstore = build_vectorstore(embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K}
        )
        flash("Index rebuilt from current PDFs.", "success")
        return redirect(url_for("index"))

    @app.route("/papers/<path:filename>")
    def serve_pdf(filename):
        return send_from_directory(DATA_DIR, filename)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
