import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama  # was Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}"),
    ]
)

st.title("Langchain Demo With Gemma Model")
input_text = st.text_input("What question you have in mind?")

# Chat model + simple guard for missing input
llm = ChatOllama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text.strip():
    st.write(chain.invoke({"question": input_text}))