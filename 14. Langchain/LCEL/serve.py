import os
from pathlib import Path
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv

# 1. SETUP ENVIRONMENT
# Using Path(__file__) is safer than Path.cwd() for scripts to ensure the .env is found
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Successfully loaded .env from: {env_path}")
else:
    # Fallback to current working directory if __file__ isn't available (like in some notebooks)
    load_dotenv()
    print("⚠️  Attempted default load_dotenv()")

# 2. INITIALIZE MODEL
groq_api_key = os.getenv("GROQ_API_KEY")

# Ensure the model is initialized with a valid, non-decommissioned ID
# llama-3.1-8b-instant is the current recommended replacement for Gemma 2
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# 3. CREATE CHAIN (LCEL)
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

# Link them together
chain = prompt_template | model | parser

# 4. APP DEFINITION
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

# 5. ADD ROUTES
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    # log_level="debug" helps you see the actual Groq errors in your terminal
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")