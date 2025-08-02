import os

from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# define llama model
llama_model = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

# define the pdf tool
pdf_tool = PDFSearchTool(pdf="rag\example_home_inspection.pdf")

# define research agent
research_agent = Agent(
    role="Research Agent",
    goal="Search through the pdf to find the relevant answers",
    backstory="""
    Research agent is good at finding and extracting information and provide relevant answers.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llama_model,
    tools=[pdf_tool]
)

# define writer agent
writer_agent = Agent(
    role="Professional Writer",
    goal="Write a professional email from the findings of the research agent",
    backstory="""
    Writer agent is good at writing professional emails and can write a professional 
    email from the findings of the research agent.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llama_model
)

# define the research task
research_tasks = Task(
    description="""
    Your job is to answer the users questions based on the home inspection report.
    Your answer must be concise, accurate and relevant and should only comes from the pdf data.

    Here are the questions:
    {questions}
    """,
    expected_output="""
    Provide clear and accurate answers to the questions based on the home inspection report.
    """,
    tools=[pdf_tool],
    agent=research_agent
)

# define the writer task
writer_task = Task(
    description="""
    - Write a professional email to the contractor based on the findings of the research agent.
    - The email should clearly state the issues found in the specified section of the report and request
        a action plan to fix the issue.
    - The email should be professional and concise and should be signed with the following details:
        Best Regards,
        Dikshant
        X Construction Company
    """,
    expected_output="""
    Write a professional email to the client based on the findings of the research agent
    to address the issues found in the home inspection report.
    """,
    agent=writer_agent
)

if __name__ == "__main__":
    crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_tasks, writer_task],
        process=Process.sequential,
        output_log_file="home_inspection.log",
    )

    customer_questions = input("Enter the questions you want to ask: ")

    result = crew.kickoff(inputs={"questions": customer_questions})

    print(result)

    # results = crew.kickoff_for_each(
    #     inputs=[
    #         {"questions": "Fan is not working"},
    #         {"questions": "Water leakage in the basement"},
    #     ],
        
    # )
