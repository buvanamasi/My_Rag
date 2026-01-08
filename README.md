# RAG Chat Application

A Retrieval Augmented Generation (RAG) application built with LangChain and Streamlit.

## Features

- 🤖 Interactive chat interface powered by Groq LLM
- 📚 Document-based Q&A using RAG
- ⚙️ Customizable settings (model, temperature, retrieval chunks)
- 💬 Chat history preservation
- 🎨 Modern and user-friendly UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. The app will automatically load and process the document from `data/Deep_Learning_basics.txt`
2. Use the sidebar to adjust settings:
   - Select different Groq models
   - Adjust temperature (0.0 = deterministic, 1.0 = creative)
   - Set number of document chunks to retrieve
3. Type your question in the chat input and press Enter
4. The app will retrieve relevant context and generate an answer

## Files

- `app.py` - Streamlit web application
- `Rag_Demo.py` - Original command-line version
- `requirements.txt` - Python dependencies
- `data/Deep_Learning_basics.txt` - Sample document for RAG

## Notes

- The vector store is cached to avoid reloading on every interaction
- Chat history is maintained during the session
- Use the "Clear Chat History" button to reset the conversation
