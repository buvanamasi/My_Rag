import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG Chat App",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD ENV ----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 RAG Chat App")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses RAG (Retrieval Augmented Generation) to answer questions based on your documents.")
    
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found in .env file!")
        st.stop()
    else:
        st.success("✅ API Key loaded")
    
    st.markdown("---")
    st.markdown("### Settings")
    model_name = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    k_retrieval = st.slider("Number of chunks to retrieve", 1, 10, 3)

# ---------------- INITIALIZE SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_settings" not in st.session_state:
    st.session_state.current_settings = {}

# ---------------- LOAD AND PROCESS DOCUMENTS ----------------
@st.cache_resource
def load_documents():
    """Load and process documents with caching"""
    with st.spinner("Loading and processing documents..."):
        # Load document
        loader = TextLoader(
            "data/Deep_Learning_basics.txt",
            encoding="utf-8"
        )
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore

# ---------------- INITIALIZE RAG CHAIN ----------------
def initialize_qa_chain(vectorstore, model_name, temperature, k_retrieval):
    """Initialize the QA chain"""
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
    
    # Create prompt template
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
    
    # Create LLM
    llm = ChatGroq(
        model=model_name,
        api_key=groq_api_key,
        temperature=temperature
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# ---------------- MAIN APP ----------------
st.title("💬 Chat with your Documents")
st.markdown("Ask questions about Deep Learning based on the loaded document.")

# Load documents
vectorstore = load_documents()

# Check if settings have changed
current_settings = {
    "model_name": model_name,
    "temperature": temperature,
    "k_retrieval": k_retrieval
}

# Initialize or reinitialize QA chain if needed
if (st.session_state.vectorstore != vectorstore or 
    st.session_state.qa_chain is None or 
    st.session_state.current_settings != current_settings):
    st.session_state.vectorstore = vectorstore
    st.session_state.current_settings = current_settings
    st.session_state.qa_chain = initialize_qa_chain(
        vectorstore, model_name, temperature, k_retrieval
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Deep Learning..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                response = result["result"]
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
