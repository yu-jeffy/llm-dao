import json
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Now you can access your API keys using os.getenv
openai_api_key = os.getenv('OPENAI_API_KEY')
# Set up environment
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {"conversation": []}

def save_history(history, file_path):
    with open(file_path, "w") as file:
        json.dump(history, file)

history_file = "history.json"
history = load_history(history_file)

def chat_with_llm(input_text, llm, history):
    response = llm.chat(input_text)
    history["conversation"].append({"timestamp": str(datetime.now()), "input": input_text, "output": response})
    save_history(history, history_file)
    return response

# Define two instances of LLMs
llm1 = ChatOpenAI()
llm2 = ChatOpenAI()

# Example function to simulate a conversation
def conduct_conversation(, turns=5):
    current_input = start_input
    for _ in range(turns):
        response = chat_with_llm(current_input, llm1, history)
        print("LLM1:", response)
        current_input = response

        response = chat_with_llm(current_input, llm2, history)
        print("LLM2:", response)
        current_input = response

# Start the conversation
conduct_conversation()