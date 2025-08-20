from typing import Annotated, TypedDict

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Streamlit page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Add a title and description
st.title("🤖 AI Chatbot")
st.markdown("Chat with me! I'm powered by GPT-4 and LangGraph.")

# Initialize the LLM
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
    checkpointer = InMemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    # compiled_graph = create_graph()
    # st.session_state.graph = compiled_graph
    thread_id = "1"
    config = {"configurable": {"thread_id": thread_id}}

    # Show a spinner while processing
    with st.spinner("Thinking..."):
        stream_response = st.session_state.graph.stream(
            {"message": [HumanMessage(content=prompt)]}, config=config, stream_mode="messages"
        )

    # Display AI response
    with st.chat_message("assistant"):
        # writes output in stream mode
        ai_response = st.write_stream(message_chunk.content for message_chunk, _ in stream_response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Add a sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This chatbot uses:
    - GPT-4.1-mini
    - LangGraph
    - Streamlit
    
    Type 'clear' to reset the chat history.
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
