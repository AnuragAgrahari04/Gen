import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text", page_icon="🦜")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    groq_api_key = st.text_input(
        "Enter your GROQ API Key:",
        value="",
        type="password",
        help="Get your API key from https://console.groq.com"
    )

    if groq_api_key:
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
        index=2,  # Default to faster model
        help="Select the Groq model for summarization"
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    max_words = st.slider(
        "Summary Length (words):",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Target length for the summary"
    )

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key above
    2. Paste a YouTube URL or website URL
    3. Click 'Summarize' to get the summary
    """)

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.write("Extract and summarize content from YouTube videos or websites using AI!")

# URL input
generic_url = st.text_input(
    "Enter YouTube URL or Website URL:",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com/article",
    help="Paste a YouTube video URL or any website URL to summarize"
)


# Function to summarize text
def summarize_content(text, llm, max_words):
    """Summarize the given text"""
    try:
        prompt_template = f"""You are an expert summarizer. Provide a clear and concise summary of the following content in approximately {max_words} words.

Content:
{text[:10000]}

Summary:"""

        response = llm.invoke(prompt_template)
        return response.content
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None


# Summarize button
if st.button("🔍 Summarize the Content", use_container_width=True, type="primary"):
    # Validate inputs
    if not groq_api_key.strip():
        st.error("❌ Please provide your GROQ API Key in the sidebar")
        st.stop()

    if not generic_url.strip():
        st.error("❌ Please enter a URL to summarize")
        st.stop()

    if not validators.url(generic_url):
        st.error("❌ Please enter a valid URL")
        st.stop()

    try:
        # Initialize LLM first
        with st.spinner("🔧 Initializing AI model..."):
            try:
                llm = ChatGroq(
                    model=model_choice,
                    groq_api_key=groq_api_key,
                    temperature=temperature
                )
                st.success("✅ Model initialized")
            except Exception as e:
                st.error(f"❌ Error initializing model: {str(e)}")
                st.info("Please check your API key")
                st.stop()

        # Load content
        with st.spinner("📥 Loading content..."):
            try:
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    st.info("🎥 Loading YouTube video...")
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False,
                        language=["en", "en-US"]
                    )
                else:
                    st.info("🌐 Loading website...")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        }
                    )

                docs = loader.load()

                if not docs or not docs[0].page_content:
                    st.error("❌ No content could be extracted from the URL")
                    st.info("Try a different URL or check if the content is publicly accessible")
                    st.stop()

                st.success(f"✅ Content loaded successfully")

                # Get text content
                full_text = "\n\n".join([doc.page_content for doc in docs])
                total_chars = len(full_text)
                st.info(f"📊 Loaded ~{total_chars:,} characters")

            except Exception as e:
                st.error(f"❌ Error loading content: {str(e)}")
                st.info("This could be due to: blocked access, invalid URL, or no transcript available (for YouTube)")
                st.stop()

        # Generate summary
        with st.spinner("🤖 Generating summary..."):
            try:
                summary = summarize_content(full_text, llm, max_words)

                if summary:
                    st.success("✅ Summary generated successfully!")
                    st.markdown("---")
                    st.subheader("📝 Summary")
                    st.write(summary)

                    # Show statistics
                    word_count = len(summary.split())
                    st.info(f"📊 Summary: {word_count} words | Original: ~{len(full_text.split())} words")

                    # Download option
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

                    # Show preview of original content
                    with st.expander("📄 Preview Original Content (first 1000 characters)"):
                        st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
                else:
                    st.error("❌ Failed to generate summary")

            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        import traceback

        st.code(traceback.format_exc())

# Example URLs
with st.expander("🎯 Example URLs to Try"):
    st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ", language="text")
    st.code("https://blog.langchain.dev/", language="text")
    st.code("https://en.wikipedia.org/wiki/Artificial_intelligence", language="text")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and Groq | 🚀 Powered by AI")