import datetime
import os
from typing import Annotated, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(find_dotenv())

# define the paths to the report
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIR_PATH = f"REPORTS/{current_time}"
os.makedirs(DIR_PATH, exist_ok=True)


def save_report(report: str) -> None:
    """Save the generated report to a markdown file.

    Args:
        report (str): The content of the report to save.
    """
    with open(f"{DIR_PATH}/report.md", "w") as f:
        f.write(report)


tools = [save_report]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class WriterState(TypedDict):
    intial_research: str
    generated_report: str
    messages: Annotated[list[str], add_messages]


def writer_node(state: WriterState):
    prompt = f"""
    You are a senior researcher tasked with writing a cohesive report for a research query. 
    You will be provided with the original query, and some initial research done by a research assistant.\n
    You should first come up with an outline for the report that describes the structure and 
    flow of the report. Then, generate the report and return that as your final output.\n
    The final output should be in markdown format, and it should be lengthy and detailed. Aim 
    for 5-10 pages of content, at least 1000 words.

    Also at the end of the report generation, save the report in the markdown format using the available tool.

    Intial Research: {state["intial_research"]}
  """

    response = llm_with_tools.invoke(prompt)

    return {"generated_report": response, "messages": [response]}


def generate_graph():
    graph = StateGraph(WriterState)
    graph.add_node("writer_assistant", writer_node)
    graph.add_node("save_report", ToolNode(tools=tools))
    graph.add_edge(START, "writer_assistant")
    graph.add_edge("writer_assistant", "save_report")
    graph.add_edge("writer_assistant", END)

    return graph.compile()


def get_final_resport(research: str):
    workflow = generate_graph()
    response = workflow.invoke({"intial_research": research})

    return response["generated_report"].tool_calls[0]["args"]["report"]
