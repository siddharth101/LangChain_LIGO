print("Importing libraries")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os
import glob

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding + LLM
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# Set up splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print("📄 Loading and processing PDFs...")
# Load and process all PDFs in a folder
all_chunks = []
pdf_files = glob.glob("/Users/siddharth/Desktop/DQ_shift/Langchains/papers/*.pdf")  # get list of all PDFs

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)  # combine all chunks from all PDFs

print("🧠 Creating vector embeddings and FAISS index...")
# Embed and create FAISS index
vectorstore = FAISS.from_documents(all_chunks, embedding)

# Build QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# Prompt user to ask questions
print("✅ Setup complete!")
print("\n🧪 LIGO Research Q&A System")
print("Ask away! (Type 'exit' to quit)")

while True:
    question = input("\n❓ Your question: ").strip()
    if question.lower() in ["exit", "quit"]:
        print("\n👋 Goodbye!")
        break

    if question == "":
        print("⚠️ Please enter a question.")
        continue

    response = qa_chain.invoke({"query": question})

    print("\n🧠 Answer:")
    print(response['result'])

    print("\n📚 Sources:")
    for doc in response['source_documents']:
        print(f"- {doc.metadata.get('source', 'unknown')}")

