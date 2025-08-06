from crewai import Agent, Task
from crewai import Crew, Process, LLM
from dotenv import load_dotenv
from obiguard import Obiguard, OBIGUARD_GATEWAY_URL

load_dotenv()

obiguard = Obiguard(provider='openai')

default_llm = LLM(
    model="openai/Qwen/Qwen2.5-32B-Instruct",
    base_url=OBIGUARD_GATEWAY_URL,
    api_key='N/A',
    extra_headers=obiguard.copy_headers()
)

# Create a researcher agent
researcher = Agent(
    role='Senior Researcher',
    goal='Discover groundbreaking technologies',
    verbose=True,
    llm=default_llm,
    backstory='A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.'
)

# Task for the researcher
research_task = Task(
    description='Identify the next big trend in AI',
    expected_output='5 paragraphs on the next big AI trend',
    agent=researcher  # Assigning the task to the researcher
)

# Instantiate your crew
tech_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential  # Tasks will be executed one after the other
)

# Begin the task execution
tech_crew.kickoff()
