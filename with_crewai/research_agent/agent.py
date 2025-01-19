import os
from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# # define llama model
# llama_model = ChatGroq(
#     model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"], temperature=0
# )

llama_model = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

# 3 different agents
"""
To create an agent we need 3 attributes:
1) Name
2) Goal
3) Backstory
"""
# planner agent
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="""
    Your're working on planning a blog article about the topic of {topic}.\
    You need to collect information that helps the audience learn something and make\
    informed decisions. Your work is the basis for the Content Writer\
    to write an article on this topic.
    """,
    allow_delegation=False,
    verbose=True,
    llm=llama_model
)

# writer agent
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
    "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
    "a new opinion piece about the topic: {topic}. "
    "You base your writing on the work of "
    "the Content Planner, who provides an outline "
    "and relevant context about the topic. "
    "You follow the main objectives and "
    "direction of the outline, "
    "as provide by the Content Planner. "
    "You also provide objective and impartial insights "
    "and back them up with information "
    "provide by the Content Planner. "
    "You acknowledge in your opinion piece "
    "when your statements are opinions "
    "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm=llama_model
)
# editor agent
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
    "from the Content Writer. "
    "Your goal is to review the blog post "
    "to ensure that it follows journalistic best practices,"
    "provides balanced viewpoints "
    "when providing opinions or assertions, "
    "and also avoids major controversial topics "
    "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=llama_model
)

# Define the task
# planner task
plan = Task(
    description=(  # what you expect the agent to do, task to be
        "1. Prioritize the latest trends, key players, "
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    # forcing function to force the agent to do the task
    expected_output="A comprehensive content plan document "
    "with an outline, audience analysis, "
    "SEO keywords, and resources.",
    agent=planner,
)

# writer task
write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
    "in markdown format, ready for publication, "
    "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

# editor task
edit = Task(
    description=(
        "Proofread the given blog post for "
        "grammatical errors and "
        "alignment with the brand's voice."
    ),
    expected_output="A well-written blog post in markdown format, "
    "ready for publication, "
    "each section should have 2 or 3 paragraphs.",
    agent=editor,
)

# Define the crew
# by default crew operates squentially
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=1
)
if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
    with open('blog_post.md', 'w') as f:
        f.write(str(result))