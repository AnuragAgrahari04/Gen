import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import re

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    api_key = st.text_input(
        "Enter your GROQ API Key:",
        type="password",
        help="Get your API key from https://console.groq.com"
    )

    if api_key:
        st.success("✅ API Key provided!")
    else:
        st.warning("⚠️ Please enter your GROQ API Key")

    st.markdown("---")

    st.header("🗄️ Database Configuration")

    radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to MySQL Database"]
    selected_opt = st.radio(
        label="Choose the database:",
        options=radio_opt
    )

    if radio_opt.index(selected_opt) == 1:
        db_uri = MYSQL
        mysql_host = st.text_input("MySQL Host", value="localhost")
        mysql_user = st.text_input("MySQL User")
        mysql_password = st.text_input("MySQL Password", type="password")
        mysql_db = st.text_input("MySQL Database")
    else:
        db_uri = LOCALDB
        mysql_host = mysql_user = mysql_password = mysql_db = None

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
        index=2,
        help="Select the Groq model for responses"
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more precise SQL, Higher = more creative"
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you query the database?"}
        ]
        st.rerun()

    st.markdown("---")

    st.header("📖 Instructions")
    st.write("""
    1. Enter your GROQ API Key
    2. Choose database (SQLite or MySQL)
    3. Ask questions in natural language
    4. Get SQL queries and results
    """)

    st.markdown("---")
    st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

# Main content
st.title("🦜 LangChain: Chat with SQL DB")
st.write("Ask questions about your database in natural language!")

# Check if API key is provided
if not api_key:
    st.error("❌ Please enter your GROQ API Key in the sidebar")
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
    st.stop()


# Configure database
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        if not dbfilepath.exists():
            st.error(f"❌ Database file not found: {dbfilepath}")
            st.stop()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("❌ Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(
            create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
        )


# Initialize database
try:
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)
    st.success(f"✅ Connected to database successfully!")
except Exception as e:
    st.error(f"❌ Database connection error: {str(e)}")
    st.stop()


# Get database schema
def get_table_info():
    """Get information about tables in the database"""
    try:
        return db.get_table_info()
    except Exception as e:
        return f"Error getting table info: {str(e)}"


# Create SQL generation prompt
sql_prompt = ChatPromptTemplate.from_template(
    """You are a SQL expert. Given the database schema and a user question, write a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}

Instructions:
1. Write ONLY the SQL query, no explanations
2. Use proper SQL syntax
3. Return only SELECT statements (no INSERT, UPDATE, DELETE)
4. Use LIMIT to restrict results if appropriate

SQL Query:"""
)

# Create answer prompt
answer_prompt = ChatPromptTemplate.from_template(
    """Based on the SQL query results, provide a clear and concise answer to the user's question.

User Question: {question}

SQL Query: {sql_query}

Query Results:
{results}

Provide a natural language answer:"""
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you query the database?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Display database info in expander
with st.expander("🗄️ View Database Schema"):
    schema_info = get_table_info()
    st.code(schema_info, language="sql")

# Chat input
if user_query := st.chat_input(placeholder="Ask anything from the database"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Generating SQL query..."):
            try:
                # Get database schema
                schema = get_table_info()

                # Generate SQL query
                sql_generation_prompt = sql_prompt.format(
                    schema=schema,
                    question=user_query
                )

                sql_response = llm.invoke(sql_generation_prompt)
                sql_query = sql_response.content.strip()

                # Clean SQL query (remove markdown code blocks if present)
                sql_query = re.sub(r'```sql\n?', '', sql_query)
                sql_query = re.sub(r'```\n?', '', sql_query)
                sql_query = sql_query.strip()

                # Display SQL query
                st.write("**Generated SQL Query:**")
                st.code(sql_query, language="sql")

                # Execute SQL query
                with st.spinner("🔍 Executing query..."):
                    try:
                        # Security check: only allow SELECT statements
                        if not sql_query.upper().strip().startswith('SELECT'):
                            st.error("❌ Only SELECT queries are allowed for security reasons")
                            raise Exception("Non-SELECT query attempted")

                        result = db.run(sql_query)

                        # Display raw results
                        st.write("**Query Results:**")
                        if result:
                            st.code(result)
                        else:
                            st.info("No results found")

                        # Generate natural language answer
                        with st.spinner("💭 Generating answer..."):
                            answer_generation_prompt = answer_prompt.format(
                                question=user_query,
                                sql_query=sql_query,
                                results=result if result else "No results found"
                            )

                            answer_response = llm.invoke(answer_generation_prompt)
                            answer = answer_response.content

                            # Display answer
                            st.write("**Answer:**")
                            st.write(answer)

                            # Save to chat history
                            full_response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Results:**\n{result}\n\n**Answer:**\n{answer}"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response
                            })

                    except Exception as e:
                        st.error(f"❌ Error executing query: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error executing query: {str(e)}"
                        })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })

# Tips section
with st.expander("💡 Example Questions"):
    st.write("""
    Try asking questions like:
    - "Show me all students"
    - "How many students are there?"
    - "What is the average grade?"
    - "List students with grades above 80"
    - "Show me the top 5 students by grade"
    - "Which subjects have the most students?"
    """)

# Show statistics
with st.expander("📊 Database Statistics"):
    try:
        tables = db.get_usable_table_names()
        st.write(f"**Number of tables:** {len(tables)}")
        st.write(f"**Tables:** {', '.join(tables)}")
    except Exception as e:
        st.error(f"Error getting statistics: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and Groq | 🚀 Powered by AI")