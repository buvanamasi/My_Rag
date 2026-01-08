from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ---------------- LOAD ENV ----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------- DATA LOADING ----------------
loader = TextLoader(
    "data/Deep_Learning_basics.txt",
    encoding="utf-8"
)
documents = loader.load()

# ---------------- TEXT SPLITTING ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- FAISS VECTOR STORE ----------------
vectorstore = FAISS.from_documents(
    texts,
    embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------- PROMPT ----------------
prompt_template = """
Use the following context to answer the question.
If you don't know the answer, say you don't know.
Use at most three sentences.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ---------------- GROQ LLM ----------------
if not groq_api_key:
    error_msg = (
        "\n" + "="*60 + "\n"
        "ERROR: GROQ_API_KEY not found!\n\n"
        "To fix this error:\n"
        "1. Create a .env file in the project root directory\n"
        "2. Add the following line to the .env file:\n"
        "   GROQ_API_KEY=your_actual_api_key_here\n"
        "3. Get your API key from: https://console.groq.com/\n"
        + "="*60 + "\n"
    )
    raise ValueError(error_msg)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0
)

# ---------------- RAG CHAIN ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# ---------------- QUESTION ANSWERING ----------------
question = "What is Deep Learning?"
result = qa_chain.invoke({"query": question})
print("\nAnswer:\n", result["result"])
