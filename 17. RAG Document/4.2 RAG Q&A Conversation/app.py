## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    # Get API key from environment or user input
    default_api_key = os.getenv("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Enter your GROQ API Key:",
        value=default_api_key,
        type="password",
        help="Get your API key from https://console.groq.com"
    )

    if api_key:
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
            "mixtral-8x7b-32768",
            "Gemma2-9b-It"
        ],
        index=0,
        help="Select the Groq model for responses"
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

    st.header("💬 Session Management")

    session_id = st.text_input(
        "Session ID:",
        value="default_session",
        help="Use different session IDs to maintain separate conversation histories"
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if 'chat_history' in st.session_state and session_id in st.session_state.chat_history:
            del st.session_state.chat_history[session_id]
            st.success("Chat history cleared!")
            st.rerun()

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key above
    2. Upload one or more PDF files
    3. Wait for processing to complete
    4. Ask questions about the content
    5. Chat history is maintained per session
    """)

    st.markdown("---")

    st.header("ℹ️ Information")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Session ID:** {session_id}")
    st.write(f"**Temperature:** {temperature}")
    st.write(f"**Embeddings:** HuggingFace")

    if 'chat_history' in st.session_state and session_id in st.session_state.chat_history:
        msg_count = len(st.session_state.chat_history[session_id])
        st.write(f"**Messages in History:** {msg_count}")

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content area
st.title("🤖 Conversational RAG With PDF Uploads")
st.write("Upload PDF files and have a conversation with their content while maintaining chat history")

# Check if API key is provided
if not api_key:
    st.error("❌ Please enter your GROQ API Key in the sidebar to continue")
    st.info("👈 Enter your API key in the sidebar")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(groq_api_key=api_key, model_name=model_choice, temperature=temperature)
except Exception as e:
    st.error(f"❌ Error initializing LLM: {str(e)}")
    st.info("Please check your API key and try again")
    st.stop()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

if session_id not in st.session_state.chat_history:
    st.session_state.chat_history[session_id] = []

# File uploader
st.subheader("📄 Upload PDF Files")
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files to chat with"
)

# Process uploaded PDFs
if uploaded_files:
    with st.spinner("📚 Processing PDF files..."):
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

            # Clean up temp file
            try:
                os.remove(temppdf)
            except:
                pass

    if documents:
        st.success(f"✅ Loaded {len(documents)} pages from {len(uploaded_files)} PDF file(s)")

        with st.spinner("🔄 Creating embeddings and vector store..."):
            # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=500
            )
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            st.success(f"✅ Created {len(splits)} document chunks")

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Chat History:
            {chat_history}

            Context from documents:
            {context}

            Question: {question}

            Answer:"""
        )

        st.markdown("---")
        st.subheader("💬 Chat with your Documents")

        # Display chat history
        if st.session_state.chat_history[session_id]:
            for message in st.session_state.chat_history[session_id]:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])

        # User input
        user_input = st.chat_input("Type your question here...")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history[session_id].append({
                'role': 'user',
                'content': user_input
            })

            # Display user message
            with st.chat_message("user"):
                st.write(user_input)

            # Get response
            with st.spinner("🤔 Thinking..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = vectorstore.similarity_search(user_input, k=4)

                    # Format context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Format chat history
                    chat_history_text = ""
                    for msg in st.session_state.chat_history[session_id][-6:]:  # Last 3 exchanges
                        if msg['role'] == 'user':
                            chat_history_text += f"User: {msg['content']}\n"
                        else:
                            chat_history_text += f"Assistant: {msg['content']}\n"

                    # Create prompt
                    formatted_prompt = prompt_template.format(
                        chat_history=chat_history_text,
                        context=context,
                        question=user_input
                    )

                    # Get response from LLM
                    response = llm.invoke(formatted_prompt)
                    answer = response.content

                    # Add assistant response to chat history
                    st.session_state.chat_history[session_id].append({
                        'role': 'assistant',
                        'content': answer
                    })

                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(answer)

                    # Show source documents in expander
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(relevant_docs):
                            st.write(f"**Source {i + 1}:**")
                            st.write(
                                doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            if 'source' in doc.metadata:
                                st.write(f"*File: {doc.metadata['source']}*")
                            st.write("---")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please try again or check your inputs")

        # Option to view full chat history
        with st.expander("📋 View Full Chat History"):
            if st.session_state.chat_history[session_id]:
                for i, msg in enumerate(st.session_state.chat_history[session_id]):
                    st.write(f"**{msg['role'].capitalize()} ({i + 1}):**")
                    st.write(msg['content'])
                    st.write("---")
            else:
                st.info("No messages in this session yet")

        # Export chat history
        if st.session_state.chat_history[session_id]:
            if st.button("💾 Export Chat History"):
                chat_text = ""
                for msg in st.session_state.chat_history[session_id]:
                    chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

                st.download_button(
                    label="📥 Download Chat as Text",
                    data=chat_text,
                    file_name=f"chat_history_{session_id}.txt",
                    mime="text/plain"
                )
    else:
        st.error("❌ No content found in uploaded PDFs")
else:
    st.info("👆 Please upload PDF files to start chatting")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and Groq")