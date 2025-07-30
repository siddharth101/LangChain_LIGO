from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os

# ✅ Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ You can pass the API key explicitly (optional if set in environment)
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# ✅ Load and split papers
loader = PyPDFLoader("path_to_papers_dir/paper1.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ✅ Embed and index
vectorstore = FAISS.from_documents(chunks, embedding)

# ✅ Create QA system
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ✅ Ask a question
response = qa_chain("What does GW170817 tell us about the neutron star equation of state?")
print(response['result'])  # To access the answer
# Optionally, print sources:
for doc in response['source_documents']:
    print(f"Source: {doc.metadata.get('source', 'unknown')}")

