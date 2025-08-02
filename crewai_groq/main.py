import time
import csv
from crewai import Crew
from agents import EmailPersonalizationAgents
from tasks import PersonalizeEmailTask

# 0. Setup environment
from dotenv import load_dotenv


load_dotenv()

email_template = """
Hey [Name]!

Just a quick reminder that we have a Skool community where you can 
join us for weekly coaching calls every Tuesday at 6 PM Eastern time.
The community is completely free and we're about to hit the 500
user milestone. We'd love to have you join us!

If you have any questions or need help with your projects, 
this is a great place to connect with others and get support. 

If you're enjoying the AI-related content, make sure to check out 
some of the other videos on my channel. Don't forget to hit that 
like and subscribe button to stay updated with the latest content. 
Looking forward to seeing you in the community!

Best regards,
Dikshant
"""

# 1. Create Agents
agents = EmailPersonalizationAgents()
email_personalization_agent = agents.personalized_email_agent()
ghostwriter_agent = agents.ghostwriter_agent()

# 2. Create Tasks
tasks = PersonalizeEmailTask()
personalize_email_tasks = []
ghostwrite_email_tasks = []

# Load data from CSV file
csv_file_path = "data\clients_large.csv"
with open(csv_file_path, mode="r", newline='') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        recipient = {
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "email": row["email"],
            "bio": row["bio"],
            "last_conversation": row["last_conversation"],
        }
        personalize_email_task = tasks.personalize_email(
            agent=email_personalization_agent,
            recipient=recipient,
            email_template=email_template,
        )
        
        ghostwrite_email_task = tasks.ghostwrite_email(
            agent=ghostwriter_agent,
            draft_email=personalize_email_task,
            recipient=recipient,
        )

        personalize_email_tasks.append(personalize_email_task)
        ghostwrite_email_tasks.append(ghostwrite_email_task)

# 3. Create Crew
crew = Crew(
    agents=[email_personalization_agent, ghostwriter_agent],
    tasks=[
        *personalize_email_tasks, 
        *ghostwrite_email_tasks
    ],
    max_rpm=20,
)

# Kick off the crew
start_time = time.time()
results = crew.kickoff()
end_time = time.time() 
print(f"Total time taken: {end_time - start_time} seconds")
print(f"Crew Usage: {crew.usage_metrics}")
