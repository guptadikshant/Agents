import subprocess
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


@tool
def create_directory(directory_name):
    """Function that creates a directory given a directory name.""" ""
    subprocess.run(["mkdir", directory_name])
    return json.dumps({"directory_name": directory_name})


@tool
def create_file(file_path):
    """Function that creates a file given a file path.""" ""
    subprocess.run(["touch", file_path])
    return json.dumps({"file_path": file_path})


llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant that helps\
                users perform tasks in the terminal.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [create_directory, create_file]

llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools)

action_input = "Create a folder called 'testing_agents' in the current directory and then create a file inside this folder called 'agents_creation.txt'"

agent_executor.invoke({"input": action_input})
