import os
import json
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Now you can access your API keys using os.getenv
openai_api_key = os.getenv('OPENAI_API_KEY')


# Define tools with explicit docstrings
@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers together and return the result."""
    return first_int + second_int

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together and return the result."""
    return first_int * second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    """Raise base to the power of exponent and return the result."""
    return base ** exponent

# Set up environment
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(temperature=0)

# Tools and Agent Setup
tools = [add, multiply, exponentiate]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a governing member of a DAO (Decentralized Autonomous Organization) and you are discussing with other members about possible proposals to implement."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_round(num_llms, initial_input, chat_history_path):
    # Start with an empty history or reset history at the start
    chat_history = []

    current_input = initial_input
    for i in range(num_llms):
        response_json = agent_executor.invoke({"input": current_input, "agent_scratchpad": chat_history})
        response = response_json['output']
        chat_history.append({"llm_id": i, "input": current_input, "response": response})
        current_input = response  # Update input for next LLM

    # Judge LLM deliberate responses
    judge_response_json = agent_executor.invoke({"input": "What do you conclude from the discussion?", "agent_scratchpad": chat_history})
    judge_response = judge_response_json['output']
    chat_history.append({"llm_id": "judge", "input": "What do you conclude from the discussion?", "response": judge_response})

    # Save history to file
    with open(chat_history_path, 'w') as f:
        json.dump(chat_history, f, indent=4)

    return judge_response

# Execute the round of conversation
result = run_round(3, "Provide a proposal to vote on and reasoning for it.", "history.json")
print(result)