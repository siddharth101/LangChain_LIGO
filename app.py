from flask import Flask, render_template, request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os
import glob

app = Flask(__name__)

# ✅ Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Init LLM & Embedding
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# ✅ Load and embed documents (only once, on app start)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []

pdf_files = glob.glob("papers/*.pdf")  # Adjust path
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)

vectorstore = FAISS.from_documents(all_chunks, embedding)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# ✅ Routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question")
        response = qa_chain(question)
        answer = response['result']
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

