import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
import re
import numexpr

# Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    groq_api_key = st.text_input(
        "Enter your GROQ API Key:",
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
            "gemma2-9b-it",
        ],
        index=0,
        help="Select the Groq model for responses"
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more precise, Higher = more creative"
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions!"}
        ]
        st.rerun()

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key above
    2. Type your math or reasoning question
    3. Click 'Find My Answer'
    4. Get detailed step-by-step solutions
    """)

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content
st.title("🧮 Text To Math Problem Solver Using AI")
st.write("Solve math problems, get Wikipedia information, and answer reasoning questions!")

# Check if API key is provided
if not groq_api_key:
    st.error("❌ Please add your GROQ API key in the sidebar to continue")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        model=model_choice,
        groq_api_key=groq_api_key,
        temperature=temperature
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {str(e)}")
    st.stop()

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()


# Function to solve math expressions
def solve_math(expression):
    """Solve mathematical expressions safely"""
    try:
        # Clean the expression
        expression = expression.strip()
        # Remove any text, keep only math
        expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        # Use numexpr for safe evaluation
        result = numexpr.evaluate(expression).item()
        return result
    except Exception as e:
        return f"Error solving math: {str(e)}"


# Function to search Wikipedia
def search_wikipedia(query):
    """Search Wikipedia for information"""
    try:
        result = wikipedia_wrapper.run(query)
        return result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# Function to determine question type and route to appropriate tool
def process_question(question, llm):
    """Process the question and determine what tools to use"""

    # First, let the LLM analyze the question
    analysis_prompt = f"""Analyze this question and determine:
1. Does it require mathematical calculation? (yes/no)
2. Does it require Wikipedia/factual information? (yes/no)
3. Is it a logic/reasoning question? (yes/no)

Question: {question}

Respond in this exact format:
Math: yes/no
Wikipedia: yes/no
Reasoning: yes/no
"""

    analysis = llm.invoke(analysis_prompt)
    analysis_text = analysis.content.lower()

    needs_math = "math: yes" in analysis_text
    needs_wikipedia = "wikipedia: yes" in analysis_text
    needs_reasoning = "reasoning: yes" in analysis_text

    return needs_math, needs_wikipedia, needs_reasoning


# Function to solve math problems with reasoning
def solve_math_problem(question, llm):
    """Solve math problems with step-by-step reasoning"""

    math_prompt = f"""You are a math expert. Solve this problem step by step.

Question: {question}

Instructions:
1. Break down the problem into steps
2. Show your calculations for each step
3. Provide the final answer clearly
4. Use clear mathematical notation

Solution:"""

    response = llm.invoke(math_prompt)
    solution = response.content

    # Try to extract and calculate any mathematical expressions
    math_expressions = re.findall(r'[\d\s+\-*/().]+(?==)', solution)

    calculations = []
    for expr in math_expressions:
        try:
            result = solve_math(expr)
            calculations.append(f"{expr.strip()} = {result}")
        except:
            pass

    if calculations:
        solution += "\n\n**Verified Calculations:**\n" + "\n".join(calculations)

    return solution


# Function to answer with reasoning
def answer_with_reasoning(question, llm, context=""):
    """Answer questions with logical reasoning"""

    reasoning_prompt = f"""You are an expert problem solver. Answer this question with clear logical reasoning.

{context}

Question: {question}

Instructions:
1. Think through the problem step by step
2. Explain your reasoning clearly
3. Provide a well-structured answer
4. Number your points for clarity

Answer:"""

    response = llm.invoke(reasoning_prompt)
    return response.content


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Question input
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
    height=100
)

# Find answer button
if st.button("🔍 Find My Answer", use_container_width=True, type="primary"):
    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing question..."):
                try:
                    # Determine what tools are needed
                    needs_math, needs_wikipedia, needs_reasoning = process_question(question, llm)

                    # Create status container
                    status_container = st.container()
                    with status_container:
                        st.write("**Processing:**")
                        if needs_math:
                            st.info("🧮 Math calculation required")
                        if needs_wikipedia:
                            st.info("📚 Searching Wikipedia")
                        if needs_reasoning:
                            st.info("🧠 Applying logical reasoning")

                    # Gather information
                    context_parts = []

                    # Search Wikipedia if needed
                    if needs_wikipedia:
                        with st.spinner("📚 Searching Wikipedia..."):
                            # Extract key terms for Wikipedia search
                            wiki_query = question.split('.')[0][:100]  # Use first sentence
                            wiki_result = search_wikipedia(wiki_query)
                            context_parts.append(f"**Wikipedia Information:**\n{wiki_result[:500]}")

                    # Solve math problem
                    if needs_math:
                        with st.spinner("🧮 Solving math problem..."):
                            math_solution = solve_math_problem(question, llm)
                            context_parts.append(f"**Mathematical Solution:**\n{math_solution}")

                    # Provide reasoning
                    if needs_reasoning or (not needs_math and not needs_wikipedia):
                        with st.spinner("🧠 Reasoning through the problem..."):
                            context = "\n\n".join(context_parts) if context_parts else ""
                            reasoning_answer = answer_with_reasoning(question, llm, context)
                            context_parts.append(f"**Detailed Explanation:**\n{reasoning_answer}")

                    # Combine all responses
                    final_response = "\n\n---\n\n".join(context_parts)

                    # Display response
                    st.markdown("### 💡 Response:")
                    st.markdown(final_response)

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    else:
        st.warning("⚠️ Please enter a question")

# Example questions
with st.expander("💡 Example Questions"):
    st.write("""
    **Math Problems:**
    - "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 berries. How many fruits do I have?"
    - "What is 25% of 480?"
    - "If a train travels at 60 mph for 2.5 hours, how far does it go?"

    **Reasoning Questions:**
    - "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
    - "A farmer has 17 sheep and all but 9 die. How many are left?"

    **Knowledge Questions:**
    - "Who invented the telephone?"
    - "What is the capital of France?"
    - "When was the Python programming language created?"
    """)

# Tips
with st.expander("🎯 Tips for Best Results"):
    st.write("""
    - **For math problems:** Be clear about the numbers and operations
    - **For word problems:** Provide all necessary information
    - **For factual questions:** Be specific about what you want to know
    - **For reasoning:** State all premises clearly

    The assistant can:
    ✅ Solve mathematical equations
    ✅ Work through word problems step-by-step
    ✅ Search Wikipedia for factual information
    ✅ Apply logical reasoning to problems
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and Groq | 🚀 Powered by AI")