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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize proposal management
proposal_number = 1
proposals = []
votes = collections.defaultdict(set)  # Track which LLMs (IDs) have voted for each proposal

@tool
def vote(proposalNumber: int, voter_id: int):
    """Increment the vote count for a proposal if the voter hasn't already voted."""
    if voter_id not in votes[proposalNumber]:
        votes[proposalNumber].add(voter_id)
        return f"Voted successfully on Proposal {proposalNumber}."
    else:
        return f"Already voted on Proposal {proposalNumber}."

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
    return f"Proposal {proposal_number-1} created successfully."

@tool
def updateProposal(proposalNumber: int, new_description: str):
    """Update the description of a specific proposal."""
    for proposal in proposals:
        if proposal["proposalNumber"] == proposalNumber:
            proposal["description"] = new_description
            return f"Proposal {proposalNumber} updated successfully."
    return f"Proposal {proposalNumber} not found."

@tool
def accessProposal(proposalNumber: int):
    """Access a proposal by its number."""
    for proposal in proposals:
        if proposal["proposalNumber"] == proposalNumber:
            return proposal
    return "Proposal not found."

@tool
def accessAllProposals():
    """Access all proposals."""
    return [f"{proposal['proposalNumber']}: {proposal['description']}" for proposal in proposals]

@tool
def speak(message: str, llm_id: int, chat_history: list):
    """Append the agent's message to the chat history."""
    chat_history.append({"llm_id": llm_id, "message": message})
    return "Message added to history."

# Agent setup
llm = ChatOpenAI(temperature=0)
tools = [createProposal, accessProposal, updateProposal, accessAllProposals, vote, speak]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a governing member of a DAO and discussing proposals."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def create_custom_prompt(dao_background, session_goal, system_prompt):
    return ChatPromptTemplate.from_messages([
        ("system", f"You are a governing member of a {dao_background}. {session_goal}"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("system", system_prompt)
    ])

def get_all_proposals():
    """Utility function to get all proposals without agent tool invocation."""
    return [f"{proposal['proposalNumber']}: {proposal['description']}" for proposal in proposals]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_round(num_llms, deliberation_rounds, initial_input, chat_history_path, dao_background, session_goal, system_prompt):
    prompt = create_custom_prompt(dao_background, session_goal, system_prompt)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chat_history = []

    for round_number in range(deliberation_rounds + 1):
        for i in range(num_llms):
            current_input = initial_input if round_number == 0 else "Please continue the discussion based on previous inputs."

            response_json = agent_executor.invoke({
                "input": current_input,
                "agent_scratchpad": json.dumps(chat_history),
                "llm_id": i
            })
            response = response_json['output']
            
            # Instead of using 'speak' tool, directly modify chat history here
            chat_history.append({"llm_id": i, "message": response})

    # Save chat history to a file
    with open(chat_history_path, 'w') as f:
        json.dump(chat_history, f, indent=4)

    return chat_history[-1]['message'], {k: len(v) for k, v in votes.items()}

# Configuration for the dynamic prompt remains the same
dao_background = """

Decentralized Autonomous Organization focused on fund management.

"""
session_goal = "The goal of this session is to deliberate and vote on the proposed measures to enhance revenue streams."
system_prompt = "Consider the impact of each proposal on our long-term objectives and operational efficiency. You can propose new proposals, update existing ones, and state suggestions."

# Test run
result, votes = run_round(3, 2, "Provide your proposals and feedback.", "history.json", dao_background, session_goal, system_prompt)
print("\nFinal Results:", result)
print("\nVotes:", votes)

