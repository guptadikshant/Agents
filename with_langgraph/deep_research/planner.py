from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

HOW_MANY_SEARCHES = 5


# defining the pydantic schemas
class WebSearchItem(BaseModel):
    reason: str
    search_query: str


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]


# Defining the State of the graph
class PlannerState(TypedDict):
    query: str
    plan: WebSearchPlan


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm = llm.with_structured_output(WebSearchPlan)


def planner_node(state: PlannerState):
    prompt = f"""
You are a helpful research assistant. Given a query, come up with a set of web searches to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.\n\n Query: {state["query"]}
"""

    result = structured_llm.invoke(prompt)

    return {"plan": result}


def create_planner_graph():
    graph = StateGraph(state_schema=PlannerState)
    graph.add_node("planner_node", planner_node)
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", END)

    return graph.compile()


def get_planned_queries(search_query: str):
    queries = []
    graph = create_planner_graph()
    graph_response = graph.invoke({"query": search_query})

    for i in range(len(graph_response["plan"].searches)):
        queries.append(graph_response["plan"].searches[i].search_query)

    return queries


# if __name__ == "__main__":
#     search_query = {
#         "query": "Search the latest trends in the AI upto best of your knowledge"
#     }

#     workflow = create_planner_graph()

#     response = workflow.invoke(search_query)

#     print(get_planned_queries(search_query=search_query['query']))

#     print(f"Response: {response['plan'].searches[1].search_query}")
