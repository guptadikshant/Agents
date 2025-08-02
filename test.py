from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

# Initialize the search tool properly wrapped as a Tool
search_tool = Tool(
    name="DuckDuckGo Search",
    description="Search the internet using DuckDuckGo",
    func=DuckDuckGoSearchRun().run
)

# Create specialized agents
travel_planner = Agent(
    role='Travel Planner',
    goal='Create detailed travel itineraries and plan logistics',
    backstory='Expert travel planner with extensive knowledge of Greece',
    tools=[search_tool],
    verbose=True
)

local_expert = Agent(
    role='Greek Local Expert',
    goal='Provide authentic local insights and recommendations',
    backstory='Native Greek with deep knowledge of local culture, customs, and hidden gems',
    tools=[search_tool],
    verbose=True
)

accommodation_expert = Agent(
    role='Accommodation Specialist',
    goal='Find and recommend the best places to stay',
    backstory='Specialized in finding the perfect accommodations based on preferences and budget',
    tools=[search_tool],
    verbose=True
)

# Create tasks
task1 = Task(
    description='Research and suggest the best time to visit Greece and create a high-level itinerary',
    agent=travel_planner,
    expected_output='A detailed analysis of the best time to visit Greece and a comprehensive day-by-day itinerary',
    output_file='greece_itinerary.txt'
)

task2 = Task(
    description='Identify must-visit locations and local experiences in Greece',
    agent=local_expert,
    expected_output='A list of recommended destinations and authentic local experiences with detailed descriptions',
    output_file='greece_destinations.txt'
)

task3 = Task(
    description='Find and recommend accommodation options in selected destinations',
    agent=accommodation_expert,
    expected_output='A curated list of accommodation options with prices, amenities, and location benefits for each destination',
    output_file='greece_accommodations.txt'
)

# Create the crew and execute the tasks
crew = Crew(
    agents=[travel_planner, local_expert, accommodation_expert],
    tasks=[task1, task2, task3],
    output_log_file='trip_planning_results.txt',
)

# Run the crew
result = crew.kickoff()

print("\nTrip Planning Results:")
print(result)