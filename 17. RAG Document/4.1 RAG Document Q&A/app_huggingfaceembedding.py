import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Read environment variables safely (don't assign None into os.environ)
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
if groq_api_key:
    os.environ['GROQ_API_KEY'] = groq_api_key
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# If GROQ key is required for this app, show a clear message and stop if missing
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set GROQ_API_KEY in your .env or environment and restart the app.")
    st.stop()

# Canonical imports for LangChain components (adjust if your environment uses different package layout)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Create a small prompt template suitable for RetrievalQA
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "Answer the question based only on the provided context.\n\n"
        "Context:\n{context}\n\nQuestion: {input}\n"
    ),
)

# Instantiate the LLM (guarded by the API key check above)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

st.title("RAG Document Q&A With Groq And Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

# Create embeddings and vector DB in session state
def create_vector_embedding():
    if "vectors" in st.session_state:
        return

    # Embeddings (HuggingFace) - will download model if not present
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Data ingestion: load PDFs from research_papers directory
    loader = PyPDFDirectoryLoader("research_papers")
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Failed to load documents from 'research_papers': {e}")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50]) if docs else []

    if not final_documents:
        st.warning("No documents were loaded or split. Make sure the 'research_papers' folder contains PDF files.")
        return

    # Build FAISS vectorstore
    try:
        vectors = FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Failed to build vector store: {e}")
        return

    # Store objects in session state
    st.session_state['embeddings'] = embeddings
    st.session_state['loader'] = loader
    st.session_state['docs'] = docs
    st.session_state['text_splitter'] = text_splitter
    st.session_state['final_documents'] = final_documents
    st.session_state['vectors'] = vectors


if st.button("Document Embedding"):
    create_vector_embedding()
    if "vectors" in st.session_state:
        st.write("Vector Database is ready")


if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Vector DB not found. Please click 'Document Embedding' to create embeddings first.")
        st.stop()

    retriever = st.session_state['vectors'].as_retriever()

    # Build a RetrievalQA chain using the prompt template
    try:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
    except Exception:
        # Fallback: try without passing the prompt (depending on langchain version)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    start = time.process_time()
    try:
        # Call the chain. Depending on LangChain version, .run, .__call__, or .invoke may be available.
        if hasattr(qa, "run"):
            answer_text = qa.run(user_prompt)
            result_sources = getattr(qa, "source_documents", None)
        else:
            result = qa({"query": user_prompt})
            # result may be a string or dict with keys 'result' or 'answer'
            if isinstance(result, dict):
                answer_text = result.get("result") or result.get("answer") or str(result)
                result_sources = result.get("source_documents") or result.get("context") or []
            else:
                answer_text = str(result)
                result_sources = []
    except Exception as e:
        st.error(f"Failed to run retrieval/QA chain: {e}")
        st.stop()

    elapsed = time.process_time() - start
    print(f"Response time :{elapsed}")

    st.write(answer_text)

    with st.expander("Document similarity Search"):
        docs = result_sources or []
        if not docs:
            st.write("No source documents returned by the chain.")
        for i, doc in enumerate(docs):
            # doc may be a Document object from LangChain
            content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
            st.write(content)
            st.write('------------------------')
