from typing import Annotated, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)


class ChatState(TypedDict):
    message: Annotated[
        list[BaseMessage], add_messages
    ]  # it contains any type of message like human message, AI message, System message


def chat_node(state: ChatState):
    messages = state["message"]
    response = llm.invoke(messages)

    return {"message": [response]}


def create_graph():
    checkpointer = MemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # intial_state = {"message": HumanMessage(content="What is the capital of India")}

    # compiled_graph = create_graph()

    # response = compiled_graph.invoke(intial_state)

    # print(f"Output: {response['message'][1].content}")
    
    compiled_graph = create_graph()
    thread_id = '1'


    while True:

        user_message = input("Human: ")
        if user_message in ['exit', 'quit', 'bye']:
            break
        
        config = {'configurable': {'thread_id': thread_id}}

        response = compiled_graph.invoke({'message': [HumanMessage(content=user_message)]}, config=config)
        print(f"AI: {response['message'][-1].content}")
