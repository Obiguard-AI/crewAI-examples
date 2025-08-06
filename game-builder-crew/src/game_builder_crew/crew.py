from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from obiguard import OBIGUARD_GATEWAY_URL


@CrewBase
class GameBuilderCrew:
    """GameBuilder crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, obiguard_client):
        self.obiguard_client = obiguard_client

    @agent
    def senior_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['senior_engineer_agent'],
            allow_delegation=False,
            verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    @agent
    def qa_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_engineer_agent'],
            allow_delegation=False,
            verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    @agent
    def chief_qa_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['chief_qa_engineer_agent'],
            allow_delegation=True,
            verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    @task
    def code_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_task'],
            agent=self.senior_engineer_agent()
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_task'],
            agent=self.qa_engineer_agent(),
            #### output_json=ResearchRoleRequirements
        )

    @task
    def evaluate_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_task'],
            agent=self.chief_qa_engineer_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the GameBuilderCrew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
