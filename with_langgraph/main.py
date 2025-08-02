import datetime
import json
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

# It's recommended to use the latest models like gpt-4o-mini or gpt-4o
llm = ChatOpenAI(model="gpt-4o-mini")

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert AI researcher. 
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries separately** for research improvements. DO NOT include them inside the reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    reflection: Reflection = Field(description="Your reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

tavily_tool = TavilySearchResults(max_results=3)


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    """Execute the tool calls"""
    last_ai_message: AIMessage = state[-1]
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        call_id = tool_call["id"]
        # Ensure the tool name is one we can handle
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            search_queries = tool_call["args"].get("search_queries", [])
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result

            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )
    return tool_messages


MAX_ITERATIONS = 2


# Renamed and corrected the routing logic function
def should_continue(state: List[BaseMessage]) -> str:
    """Decide whether to continue the revision loop or end."""
    # Count how many tool executions we've done.
    num_tool_messages = sum(isinstance(msg, ToolMessage) for msg in state)
    # If we have reached the maximum number of iterations, end.
    if num_tool_messages >= MAX_ITERATIONS:
        return "end"
    # Otherwise, continue the loop.
    return "continue"


# Define the graph
graph = MessageGraph()
graph.set_entry_point("initial_response")
graph.add_node(
    "initial_response", lambda state: first_responder_chain.invoke({"messages": state})
)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", lambda state: revisor_chain.invoke({"messages": state}))
graph.add_edge("initial_response", "execute_tools")
graph.add_edge("execute_tools", "revisor")
graph.add_conditional_edges(
    "revisor",
    event_loop,
    {
        "execute_tools": "execute_tools",  # Corrected routing
        END: END,
    },
)

app = graph.compile()

# Invoke the app with the corrected input format
response = app.invoke(
    {
        "messages": [
            ("human", "Write about how small businesses can leverage AI to grow")
        ]
    }
)

# You can print the final response to see the result
print(response[-1].content)
