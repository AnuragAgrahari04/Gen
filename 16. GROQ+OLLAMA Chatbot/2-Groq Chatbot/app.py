import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# 1. Load the variables from your .env file
load_dotenv() 

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Groq"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, model_name, temperature, max_tokens):
    # 2. Use the GROQ_API_KEY from the environment
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With Groq")

## Sidebar for settings
st.sidebar.title("Settings")

# 3. Automatically fetch the key from .env using your specific name
# If the key is not in .env, it will show as an empty field in the sidebar
saved_api_key = os.getenv("GROQ_API_KEY") 
api_key = st.sidebar.text_input("Enter your Groq API Key:", value=saved_api_key, type="password")

## Select the Groq model
model_name = st.sidebar.selectbox(
    "Select Model", 
    ["llama-3.1-8b-instant"]
)

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1024, value=150)

## Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    try:
        response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif user_input:
    st.warning("Please ensure GROQ_API_KEY is set in your .env file or sidebar.")
else:
    st.write("Please provide the user input")