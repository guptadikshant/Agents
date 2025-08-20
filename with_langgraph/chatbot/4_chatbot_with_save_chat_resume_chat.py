import uuid
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


def generate_thread_id():
    """
    Generate a random thread id for each new conversation
    Returns:
        None: returns None
    """
    return uuid.uuid4()


def reset_chat():
    """
    Reset the chat state.
    """
    st.session_state.thread_id = generate_thread_id()
    add_thread(st.session_state.thread_id)
    st.session_state.messages = []
    st.rerun()


def add_thread(thread_id):
    """
    Add a new thread to the chat.
    """
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


class ChatState(TypedDict):
    """
    Class defines the state of the graph
    Args:
        TypedDict (dict): A dictionary that defines the structure of the chat state
    """

    message: Annotated[
        list[BaseMessage], add_messages
    ]  # it contains any type of message like human message, AI message, System message


def chat_node(state: ChatState):
    """
    Process chat messages and generate a response.
    Args:
        state (ChatState): The current state of the chat.

    Returns:
        dict: A dictionary containing the generated response.
    """
    messages = state["message"]
    response = llm.invoke(messages)

    return {"message": [response]}


def create_graph():
    """
    Create a state graph for the chat.

    Returns:
        StateGraph: The state graph for the chat.
    """
    checkpointer = InMemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


def load_conversations(thread_id):
    """
    Load conversations for a specific thread.
    """
    try:
        state = st.session_state.graph.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        if state and "message" in state.values:  # Changed from 'messages' to 'message'
            return state.values["message"]
        return []
    except Exception:
        return []


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

add_thread(st.session_state.thread_id)


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
    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # Show a spinner while processing
    with st.spinner("Thinking..."):
        stream_response = st.session_state.graph.stream(
            {"message": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="messages",
        )

    # Display AI response
    with st.chat_message("assistant"):
        # writes output in stream mode
        ai_response = st.write_stream(
            message_chunk.content for message_chunk, _ in stream_response
        )

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

    if st.button("New Chat"):
        reset_chat()
    st.markdown("My Conversations")

    for thread_ids in st.session_state.chat_threads:
        if st.button(f"{thread_ids}"):
            st.session_state.thread_id = thread_ids
            messages = load_conversations(thread_ids)
            if messages:
                temp_msg = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role = "user"
                    else:
                        role = "assistant"
                    temp_msg.append({"role": role, "content": msg.content})
                st.session_state.messages = temp_msg
            