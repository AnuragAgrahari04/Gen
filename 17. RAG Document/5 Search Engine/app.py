import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Sidebar for settings
with st.sidebar:
    st.header("🔑 API Configuration")

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

    model_choice = st.selectbox(
        "Choose Model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        index=2,  # Default to 8b-instant for speed
        help="Select the Groq model for responses"
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    st.markdown("---")

    st.header("🔍 Search Tools")

    use_search = st.checkbox("Web Search (DuckDuckGo)", value=True)
    use_arxiv = st.checkbox("Academic Papers (Arxiv)", value=True)
    use_wiki = st.checkbox("Wikipedia", value=True)

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key above
    2. Select which search tools to use
    3. Ask any question
    4. The AI will search and provide answers
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]
        st.rerun()

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content
st.title("🔎 LangChain - Chat with Search")
st.write("""
This AI assistant can search the web, academic papers, and Wikipedia to answer your questions.
Ask anything and I'll find the most relevant information for you!
""")

# Check if API key is provided
if not api_key:
    st.error("❌ Please enter your GROQ API Key in the sidebar to continue")
    st.info("👈 Enter your API key in the sidebar")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_choice,
        temperature=temperature
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {str(e)}")
    st.info("Please check your API key and try again")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg['content'])


# Function to search using available tools
def search_with_tools(query):
    results = []

    # Web Search
    if use_search:
        try:
            with st.status("🔍 Searching the web...", expanded=False):
                search_result = search.run(query)
                results.append({"source": "Web Search", "content": search_result})
                st.write(f"✅ Found web results")
        except Exception as e:
            st.write(f"⚠️ Web search failed: {str(e)}")

    # Arxiv Search
    if use_arxiv:
        try:
            with st.status("📚 Searching academic papers...", expanded=False):
                arxiv_result = arxiv.run(query)
                results.append({"source": "Arxiv", "content": arxiv_result})
                st.write(f"✅ Found academic papers")
        except Exception as e:
            st.write(f"⚠️ Arxiv search failed: {str(e)}")

    # Wikipedia Search
    if use_wiki:
        try:
            with st.status("📖 Searching Wikipedia...", expanded=False):
                wiki_result = wiki.run(query)
                results.append({"source": "Wikipedia", "content": wiki_result})
                st.write(f"✅ Found Wikipedia articles")
        except Exception as e:
            st.write(f"⚠️ Wikipedia search failed: {str(e)}")

    return results


# Create prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant with access to web search, academic papers, and Wikipedia.

    User Question: {question}

    Search Results:
    {search_results}

    Based on the search results above, provide a comprehensive and accurate answer to the user's question.
    If the search results don't contain relevant information, say so honestly.
    Always cite your sources by mentioning which tool provided the information (Web Search, Arxiv, or Wikipedia).

    Answer:"""
)

# Chat input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and searching..."):
            try:
                # Search using tools
                search_results = search_with_tools(prompt)

                # Format search results
                formatted_results = ""
                for i, result in enumerate(search_results, 1):
                    formatted_results += f"\n{i}. {result['source']}:\n{result['content']}\n"

                if not formatted_results:
                    formatted_results = "No search results found."

                # Create prompt with search results
                formatted_prompt = prompt_template.format(
                    question=prompt,
                    search_results=formatted_results
                )

                # Get LLM response
                response = llm.invoke(formatted_prompt)
                answer = response.content

                # Display answer
                st.write(answer)

                # Show search results in expander
                with st.expander("🔍 View Search Results"):
                    for result in search_results:
                        st.write(f"**{result['source']}:**")
                        st.write(result['content'])
                        st.write("---")

                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please try again or check your inputs")

# Show tips
with st.expander("💡 Tips for Better Results"):
    st.write("""
    - **For academic questions:** Enable Arxiv search
    - **For general knowledge:** Enable Wikipedia
    - **For current events:** Enable Web Search
    - **For best results:** Enable all three tools

    **Example questions:**
    - "What is quantum computing?"
    - "Latest developments in AI research"
    - "Explain the theory of relativity"
    - "Current weather in New York"
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and Groq | 🚀 Powered by AI")