import os
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from pathlib import Path # Import Path for robust path handling

# Load environment variables from .env file (ONLY FOR LOCAL DEVELOPMENT)
# On deployment platforms (Streamlit Cloud, Hugging Face Spaces), set GEMINI_API_KEY as a secret
# load_dotenv() # <--- REMOVE THIS LINE for production deployment

# Ensure the Gemini API key is available
# This check now relies *solely* on the environment variable being set
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Gemini API key not found. Please set it as an environment variable (GEMINI_API_KEY).")
    st.stop() # Stop the Streamlit app execution if the key is missing

INDEX_DIR = 'wiki-rag'
DEFAULT_PAGES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep Learning",
    "Neural network",
    "Convolutional neural network",
    "Reinforcement learning",
    "Supervised learning"
]

@st.cache_resource
def get_index(selected_pages):
    # Create a unique cache key based on selected pages
    # Use Path for better cross-OS path handling
    cache_key = "_".join(sorted(selected_pages)).replace(" ", "_").lower() # Make key filesystem friendly
    index_path = Path(INDEX_DIR) / cache_key
    
    try:
        if index_path.is_dir():
            st.info(f"Loading existing index from {index_path}...")
            storage = StorageContext.from_defaults(persist_dir=str(index_path))
            return load_index_from_storage(storage)
    except Exception as e:
        st.warning(f"Failed to load existing index from {index_path}: {e}. Creating a new index...")

    # If index not loaded or failed, create a new one
    try:
        st.info(f"Fetching data from Wikipedia for pages: {', '.join(selected_pages)}...")
        docs = WikipediaReader().load_data(pages=selected_pages, auto_suggest=False)
        
        if not docs:
            st.error("No documents found for the specified Wikipedia pages. Please check the page names.")
            return None
            
        # Use a local embedding model
        st.info("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        st.info("Creating vector store index...")
        index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
        
        st.info(f"Persisting index to {index_path}...")
        index.storage_context.persist(persist_dir=str(index_path))
        
        return index
        
    except Exception as e:
        st.error(f"Error creating index: {e}. Please ensure Wikipedia pages exist and try again.")
        return None

@st.cache_resource
def get_query_engine(selected_pages):
    index = get_index(selected_pages)
    if index is None:
        return None
        
    # Use Gemini API for the LLM
    llm = Gemini(
        model_name="models/gemini-1.5-flash", 
        api_key=gemini_api_key, # Use the directly retrieved key
        temperature=0.7
    )
    
    st.info("Initializing query engine...")
    # Consider adjusting similarity_top_k based on your needs.
    # similarity_top_k=1 might be too restrictive; often 2-4 is better for RAG.
    return index.as_query_engine(llm=llm, similarity_top_k=2) # Changed to 2 for potentially better context

def main():
    st.set_page_config(page_title="Wikipedia RAG App", page_icon="📚", layout="centered")
    st.title('📚 Wikipedia RAG Application')
    
    st.markdown("""
    This application uses LlamaIndex to query Wikipedia pages.
    It builds a local index for the selected pages using a HuggingFace embedding model,
    and then uses Google Gemini for answering questions based on the retrieved context.
    """)

    # Allow users to select pages to reduce initial load
    st.subheader("Select Wikipedia Pages to Query")
    
    # Use a unique key for the multiselect to avoid potential caching issues on widget changes
    selected_pages = st.multiselect(
        "Choose pages (fewer pages = faster response and indexing)",
        options=DEFAULT_PAGES,
        default=["Machine learning"],
        key="wiki_page_selection"
    )
    
    # Clear cache button for debugging or re-indexing (optional)
    if st.sidebar.button("Clear Cache and Re-index"):
        st.cache_resource.clear()
        st.success("Cache cleared! Please re-select pages or refresh the app to re-index.")
        st.experimental_rerun() # Rerun the app to re-initialize

    if not selected_pages:
        st.warning("Please select at least one Wikipedia page to get started.")
        return # Stop execution if no pages are selected
        
    # Initialize query engine with selected pages
    qa = None
    with st.spinner(f"Indexing {len(selected_pages)} selected pages... This may take a moment."):
        qa = get_query_engine(selected_pages)
        if qa is None:
            st.error("Failed to initialize the query engine. Please check the logs above.")
            return

    st.success("Query engine ready! You can now ask questions.")

    # Query input
    question = st.text_input('Ask a question about the selected Wikipedia pages:', 
                             placeholder="e.g., What are the main types of machine learning?",
                             key="user_question_input")
    
    if st.button('Get Answer', key="submit_button"):
        if not question:
            st.warning("Please enter a question before submitting.")
            return
            
        with st.spinner('Thinking and retrieving information...'):
            try:
                response = qa.query(question)
                
                st.subheader('Answer')
                st.write(response.response)
                
                if response.source_nodes:
                    st.subheader('Retrieved Context (from Wikipedia)')
                    for i, src in enumerate(response.source_nodes):
                        # Displaying only a portion of the text to avoid overwhelming the UI
                        content = src.node.get_content().strip()
                        st.expander(f"Source Document {i+1} (Score: {src.score:.2f}) - Page: {src.node.metadata.get('title', 'N/A')}") \
                          .text(content[:500] + "..." if len(content) > 500 else content)
                else:
                    st.info("No specific context was retrieved for this query. The answer might be general knowledge.")

            except Exception as e:
                st.error(f"Error processing query: {e}. Please try a different question or check the console for details.")
                # You might want to log the full traceback in a real deployment
                # import traceback
                # st.exception(e) # This will show the traceback in Streamlit

if __name__ == '__main__':
    main()
