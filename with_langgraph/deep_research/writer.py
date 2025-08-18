from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class WriterState(TypedDict):
    intial_research: str
    generated_report: str


def writer_node(state: WriterState):
    prompt = f"""
    You are a senior researcher tasked with writing a cohesive report for a research query. 
    You will be provided with the original query, and some initial research done by a research assistant.\n
    You should first come up with an outline for the report that describes the structure and 
    flow of the report. Then, generate the report and return that as your final output.\n
    The final output should be in markdown format, and it should be lengthy and detailed. Aim 
    for 5-10 pages of content, at least 1000 words.

    Intial Research: {state["intial_research"]}
  """

    response = llm.invoke(prompt).content

    return {"generated_report": response}


def generate_graph():
    graph = StateGraph(WriterState)
    graph.add_node("writer_assistant", writer_node)
    graph.add_edge(START, "writer_assistant")
    graph.add_edge("writer_assistant", END)

    return graph.compile()


def get_final_resport(research: str):
    workflow = generate_graph()
    response = workflow.invoke({"intial_research": research})

    return response['generated_report']
