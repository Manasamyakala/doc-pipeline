# RAG PDF Explorer

This is a Retrieval-Augmented Generation (RAG) application that answers user questions using the content of uploaded PDF documents. It is built to satisfy the assignment requirements, utilizing Google Gemini for LLMs and Embeddings, and FAISS for the vector database.

## Architecture
The application follows this workflow:
1. **Upload**: Users upload one or more PDF files via the Streamlit UI.
2. **Extract & Split**: Text is extracted using `PyPDFLoader` and split into chunks using `RecursiveCharacterTextSplitter`. Chunk size and overlap are configurable via the sidebar.
3. **Embeddings & Vector DB**: The chunks are embedded using Google's `gemini-embedding-2-preview` model and stored in a FAISS vector database.
4. **Retrieval**: When a user asks a question, the top 5 most relevant chunks are retrieved.
5. **QA**: A prompt containing the retrieved context and user question is sent to `gemini-2.5-flash` to generate a concise answer strictly based on the context.

## Libraries Used
- **Streamlit**: Web interface and UI components.
- **Langchain**: Orchestration of the RAG pipeline.
- **PyPDF**: Extracting text from PDF documents.
- **Google Generative AI**: LLM (`gemini-2.5-flash`) and Embeddings.
- **FAISS**: In-memory vector database for fast similarity search.

## Setup Instructions

1. **Clone or Download the Repository**
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
4. **Usage**
   - Provide your Google API key in the sidebar.
   - Upload one or more PDFs.
   - Click "Process Documents".
   - Start asking questions in the chat interface!

## Challenges Faced (Placeholder for Report)
- *Handling in-memory PDF uploads with Streamlit and passing them to Langchain's document loaders required writing them to temporary files.*
- *Adapting the provided notebook (which used simple text files) to handle PDF processing and metadata attribution for source tracking.*

## Future Improvements (Placeholder for Report)
- Implementing persistent storage for the FAISS database so users don't have to re-process documents on refresh.
- Adding hybrid search capabilities (combining keyword/BM25 with vector search).
- Extracting tables and images from PDFs for multimodal RAG.
