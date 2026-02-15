import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    Context: {context}

    Question: {question}

    Answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector embeddings... This may take a few minutes."):
            # Using HuggingFace embeddings (free, no installation needed)
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()

            if not st.session_state.docs:
                st.error("No PDF files found in research_papers folder!")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )


# Sidebar with API Key input and settings
with st.sidebar:
    st.header("🔑 API Configuration")

    # Get API key from environment or user input
    default_api_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = st.text_input(
        "Enter your GROQ API Key:",
        value=default_api_key,
        type="password",
        help="Get your API key from https://console.groq.com"
    )

    if groq_api_key:
        st.success("✅ API Key provided!")
    else:
        st.warning("⚠️ Please enter your GROQ API Key")

    st.markdown("---")

    st.header("⚙️ Model Settings")

    # Model selector
    model_choice = st.selectbox(
        "Choose Model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        help="llama-3.3-70b-versatile is recommended for best results"
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key above
    2. Add PDF files to the `research_papers` folder
    3. Click 'Create Embeddings' button
    4. Wait for processing to complete
    5. Ask your questions in the text box
    """)

    st.markdown("---")

    st.header("ℹ️ Information")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Embeddings:** HuggingFace")
    st.write(f"**Temperature:** {temperature}")

    if "vectors" in st.session_state:
        st.write(f"**Document Chunks:** {len(st.session_state.final_documents)}")

    st.markdown("---")

    if st.button("🔄 Reset Database", use_container_width=True):
        if "vectors" in st.session_state:
            del st.session_state.vectors
            del st.session_state.docs
            del st.session_state.final_documents
            st.success("Database reset successfully!")
            st.rerun()

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content area
st.title("RAG Document Q&A With Groq And Llama3")
st.write("Upload your PDF research papers to the 'research_papers' folder and ask questions!")

# Check if API key is provided
if not groq_api_key:
    st.error("❌ Please enter your GROQ API Key in the sidebar to continue")
    st.info("👈 Enter your API key in the sidebar")
    st.stop()

# Initialize LLM with user's API key
try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_choice,
        temperature=temperature
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {str(e)}")
    st.info("Please check your API key and try again")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    user_prompt = st.text_input("Enter your query from the research paper")

with col2:
    st.write("")  # Spacer
    if st.button("📚 Create Embeddings", use_container_width=True):
        create_vector_embedding()
        st.success("✅ Vector Database is ready!")

# Show status
if "vectors" in st.session_state:
    st.info(f"📊 Database contains {len(st.session_state.final_documents)} document chunks")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first by clicking the 'Create Embeddings' button.")
    else:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Use similarity_search
                relevant_docs = st.session_state.vectors.similarity_search(user_prompt, k=4)

                # Create the context from relevant documents
                context = format_docs(relevant_docs)

                # Format the prompt
                formatted_prompt = prompt.format(context=context, question=user_prompt)

                # Get response from LLM
                start = time.process_time()
                response = llm.invoke(formatted_prompt)
                response_time = time.process_time() - start

                # Display results
                st.write(f"⏱️ **Response time:** {response_time:.2f} seconds")
                st.write("---")
                st.write("### 💡 Answer:")
                st.write(response.content)

                # Document similarity search
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(relevant_docs):
                        st.write(f"**Document {i + 1}:**")
                        st.write(doc.page_content)
                        st.write(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                        st.write('---')

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your API key and try again")