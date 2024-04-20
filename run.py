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
import collections

# Loading environment variables from the .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

# Setting up a counter for proposal votes using defaultdict
votes = collections.defaultdict(int)


@tool
def vote(proposalNumber: int):
    """Increment the vote count for a proposal."""
    votes[proposalNumber] += 1


proposal_number = 1

proposals = []


@tool
def createProposal(proposal: str):
    """Create a proposal for the DAO."""
    global proposal_number
    proposal_data = {
        "proposalNumber": proposal_number,
        "description": proposal
    }
    proposals.append(proposal_data)
    proposal_number += 1
    return proposal

@tool
def accessProposal(proposalNumber: int):
    """Access a proposal by its number."""
    for proposal in proposals:
        if proposal["proposalNumber"] == proposalNumber:
            return proposal["description"]
    return "Proposal not found."

@tool
def accessAllProposals():
    """Access all proposals."""
    return [f"{proposal['proposalNumber']}: {proposal['description']}" for proposal in proposals]

# Tools and Agent Setup
tools = [vote]
llm = ChatOpenAI(temperature=0)

# Creating a prompt for our DAO scenario
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a governing member of a DAO (Decentralized Autonomous Organization) and you are discussing with other members about possible proposals to implement."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to handle chat history directly


def run_round(num_llms, deliberation_rounds, initial_input, chat_history_path):
    chat_history = []

    # Initial proposals round
    current_input = initial_input
    print("\nStarting initial proposal round...")
    for i in range(num_llms):
        response_json = agent_executor.invoke(
            {"input": current_input, "agent_scratchpad": json.dumps(chat_history)})
        response = response_json['output']
        chat_history.append(
            {"llm_id": i, "input": current_input, "response": response})
        current_input = response  # Update input for next LLM

    # Deliberation rounds
    for round_number in range(1, deliberation_rounds + 1):
        print(f"\nDeliberation round {round_number}...")
        for i in range(num_llms):
            response_json = agent_executor.invoke(
                {"input": f"Discuss and build upon previous proposals.", "agent_scratchpad": json.dumps(chat_history)})
            response = response_json['output']
            chat_history.append(
                {"llm_id": i, "input": "Discuss and build upon previous proposals.", "response": response})
            current_input = response

    # Voting round
    print("\nStarting voting round...")
    for i in range(num_llms):
        response_json = agent_executor.invoke(
            {"input": "Please vote on the proposals discussed.", "agent_scratchpad": json.dumps(chat_history)})
        response = response_json['output']
        chat_history.append(
            {"llm_id": i, "input": "Please vote on the proposals discussed.", "response": response})

    # Saving chat history to file
    with open(chat_history_path, 'w') as f:
        json.dump(chat_history, f, indent=4)

    return chat_history[-1]['response'], votes


# Execute the round of conversation
deliberation_rounds = 1  # Change as needed
result, votes = run_round(3, deliberation_rounds, "Comment on all existing proposals, providing feedback and analysis. Then, provide a proposal to vote on and reasoning for it. Number your proposals and refer to others as with their numbers.", "history.json")
print("\nFinal Results:", result)
print("\nVotes:", dict(votes))
