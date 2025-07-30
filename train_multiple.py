from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os
import glob

# ✅ Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize embedding + LLM
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# ✅ Set up splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# ✅ Load and process all PDFs in a folder
all_chunks = []
pdf_files = glob.glob("/Users/siddharth/Desktop/DQ_shift/Langchains/papers/*.pdf")  # get list of all PDFs

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)  # combine all chunks from all PDFs

# ✅ Embed and create FAISS index
vectorstore = FAISS.from_documents(all_chunks, embedding)

# ✅ Build QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ✅ Ask your question
question = "Tell me about GW170817"
question = "Tell me about LIGO detectors"
response = qa_chain(question)
print(response['result'])

# ✅ Print sources
for doc in response['source_documents']:
    print(f"Source: {doc.metadata.get('source', 'unknown')}")

