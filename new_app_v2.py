import os
from pathlib import Path
from dotenv import load_dotenv
import time
import unicodedata, re

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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate

# ------------- Config -------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
DATA_DIR = Path("papers")
INDEX_DIR = Path("index/faiss")
ALLOWED_EXTS = {".pdf"}


from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate

def normalize_text(s: str) -> str:
    if s is None:
        return s
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("p astro", "pastro").replace("pₐₛₜᵣₒ", "pastro")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_ensemble_retriever(vectorstore: FAISS) -> EnsembleRetriever:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    all_chunks = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            d.page_content = normalize_text(d.page_content)
            d.metadata = d.metadata or {}
            d.metadata["source"] = pdf_path.name
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = 8
    faiss_ret = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 48})
    return EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.6, 0.4])

def make_qa_chain(llm, retriever, prompt_text: str | None = None) -> RetrievalQA:
    if prompt_text is None:
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    prompt = PromptTemplate(input_variables=["context"], template=prompt_text)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt})

PASTRO_PROMPT_TXT = """Answer using only the context. If multiple counts appear, list each as:
- <count> — <criterion/filters> — <source filename>
Then select the count that matches: "identified by at least one search algorithm, pastro ≥ 0.5, and not vetoed during event validation" and label it **Final**.
Context:
{context}
Answer:"""

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
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

    # ---- Vectorstore & retriever ----
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = build_ensemble_retriever(vectorstore)

    # Default QA chain; we’ll swap prompts dynamically for “pastro” queries
    qa_chain = make_qa_chain(llm, retriever)

    @app.route("/", methods=["GET", "POST"])
    def index():
        answer, sources, question = "", [], ""
        if request.method == "POST":
            question = (request.form.get("question") or "").strip()
                      # --- Conditional, group-based query expansion ---
            ql = question.lower()
            expanded_parts = [question]

            # Group A: pastro synonyms
            PASTRO_TRIGGERS = ["pastro", "probability of astrophysical origin", "astroprob", "p_astro"]
            PASTRO_EXPAND = [
                "probability of astrophysical origin ≥ 0.5",
                "pastro ≥ 0.5",
                "probability of astrophysical origin > 0.5"
            ]
            if any(t in ql for t in PASTRO_TRIGGERS):
                expanded_parts.append(" ".join(PASTRO_EXPAND))

            # Group B: O4a / fourth observing run synonyms
            O4_TRIGGERS = ["o4a", "o4", "fourth observing run", "first half of fourth observing run"]
            O4_EXPAND = ["O4a", "first half of fourth observing run", "fourth observing run"]
            if any(t in ql for t in O4_TRIGGERS):
                expanded_parts.append(" ".join(O4_EXPAND))

            # Group C: detection wording
            DET_TRIGGERS = ["detection", "detections", "detected", "gravitational wave events", "gravitational-wave events"]
            DET_EXPAND = ["detections", "gravitational wave detections"]
            if any(t in ql for t in DET_TRIGGERS):
                expanded_parts.append(" ".join(DET_EXPAND))

            expanded_query = " ".join(expanded_parts)
            if question:
                chain = qa_chain
                if "pastro" in question.lower():
                    chain = make_qa_chain(llm, retriever, PASTRO_PROMPT_TXT)

                t0 = time.perf_counter()
                result = chain({"query": expanded_query})
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                answer = result.get("result", "")
                src_docs = result.get("source_documents", []) or []
                for d in src_docs:
                    snippet = (d.page_content or "")[:400].strip().replace("\n", " ")
                    sources.append({
                        "filename": (d.metadata or {}).get("source", "unknown.pdf"),
                        "snippet": snippet + ("..." if d.page_content and len(d.page_content) > 400 else "")
                    })

                usage = result.get("generation_info") or {}
                prompt_toks = usage.get("prompt_tokens")
                completion_toks = usage.get("completion_tokens")
                total_toks = usage.get("total_tokens")

                retriever_cfg = (
                    f"type={getattr(retriever, 'search_type', 'ensemble')};"
                    f"k=8;fetch_k=48"
                )

                # Safe logger (if json_logger missing)
                try:
                    write_json_log(
                        question=question, answer=answer, latency_ms=elapsed_ms,
                        model=getattr(llm, "model_name", getattr(llm, "model", None)),
                        embedding_model=getattr(embeddings, "model", None),
                        retriever_cfg=retriever_cfg,
                        prompt_tokens=prompt_toks, completion_tokens=completion_toks, total_tokens=total_toks,
                        client_ip=request.headers.get("X-Forwarded-For", request.remote_addr),
                        user_agent=request.headers.get("User-Agent"),
                        source_documents=src_docs,
                    )
                except Exception:
                    pass

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
        nonlocal vectorstore, retriever, qa_chain
        vectorstore = build_vectorstore(embeddings)
        retriever = build_ensemble_retriever(vectorstore)
        qa_chain = make_qa_chain(llm, retriever)
        flash("Index rebuilt from current PDFs.", "success")
        return redirect(url_for("index"))

    @app.route("/papers/<path:filename>")
    def serve_pdf(filename):
        return send_from_directory(DATA_DIR, filename)

    return app

if __name__ == "__main__":
    import traceback
    try:
        app = create_app()
        assert app is not None, "create_app() returned None"
    except Exception:
        traceback.print_exc()
        raise
    app.run(debug=True)
