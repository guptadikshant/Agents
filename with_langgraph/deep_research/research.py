import json
from typing import TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tool = [TavilySearchResults(max_results=5, topic="general")]
llm_with_tool = llm.bind_tools(tool)


class ResearchState(TypedDict):
    research_topic: str
    messages: list


def search_topic(state: ResearchState):
    prompt = f"""
You are a intelligent research assistant which searches on the internet on a below given topic. The search result should be of 2-3 lines summary and less than 300 words. Capture the main points, no need to capture the whole sentences. This summary will be consumed by someone who will creates final report out of it. So it is vital for you to capture the essence of the topic and ignore any fluff.

NOTE: If you are not able to run the available tool(s), the reply with best of your knowledge

\n\n Search Topic: {state["research_topic"]}
"""
    response = llm_with_tool.invoke(prompt)

    return {"searched_output": response.content, "messages": [response]}


def create_search_graph():
    graph = StateGraph(state_schema=ResearchState)
    graph.add_node("search_agent", search_topic)
    graph.add_node("tools", ToolNode(tools=tool))
    graph.add_edge(START, "search_agent")
    graph.add_edge("search_agent", "tools")
    graph.add_conditional_edges("search_agent", tools_condition)
    graph.add_edge("search_agent", END)

    return graph.compile()


def get_search_responses(research_topic: str):
    workflow = create_search_graph()
    graph = workflow.invoke({"research_topic": research_topic})

    return json.loads(graph["messages"][0].content)[0]["content"]


# if __name__ == "__main__":
#     workflow = create_search_graph()
#     messages = workflow.invoke({"research_topic": "GPT-5 Launch"})
#     print(f"Response:{json.loads(messages['messages'][0].content)[0]['content']}")
