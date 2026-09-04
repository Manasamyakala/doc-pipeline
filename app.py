import streamlit as st

from src.document_processor import load_and_split_pdfs
from src.rag_pipeline import create_vector_store, setup_rag_chain, generate_answer


st.set_page_config(page_title="DocuMind AI", page_icon="📚")
st.title(" DocuMind AI: RAG PDF Explorer")

# Initialize session state for chat history and RAG components
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for inputs and settings
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    api_key = st.text_input(
        "Google API Key", 
        type="password", 
        value="", # User must enter their own key
        help="Required to use Google Gemini models."
    )
    
    st.markdown("---")
    
    # File Uploader
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    
    # Chunk Settings
    st.header("2. Text Processing Settings")
    chunk_size = st.slider("Chunk Size", min_value=50, max_value=1000, value=200, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=2, step=1)
    
    # Process Button
    if st.button("Process Documents"):
        if not api_key:
            st.error("Please provide a Google API Key.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents..."):
                # 1 & 2: Load and Split
                chunks = load_and_split_pdfs(uploaded_files, chunk_size, chunk_overlap)
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks.")
                
                # 3 & 4: Embeddings and Vector DB
                with st.spinner("Creating Vector Database..."):
                    try:
                        vector_store = create_vector_store(chunks, api_key)
                        st.session_state.vector_store = vector_store
                        
                        # 5 & 6: Setup RAG Chain
                        st.session_state.rag_chain = setup_rag_chain(vector_store, api_key)
                        
                        st.success("Vector Database created and RAG Chain initialized! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error during vector store creation: {e}")

# Main chat area
st.subheader("Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your uploaded documents"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.rag_chain is None:
        response = "Please upload documents and process them first."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Generate Answer
                    response = generate_answer(st.session_state.rag_chain, prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Bonus feature: Retrieve source pages if possible
                    # We can manually retrieve to show sources
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(prompt)
                    
                    with st.expander("View Source Chunks"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1}** (from `{doc.metadata.get('source_file', 'Unknown')}`):")
                            st.text(doc.page_content)
                            st.markdown("---")
                            
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.error(response)
                    
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
