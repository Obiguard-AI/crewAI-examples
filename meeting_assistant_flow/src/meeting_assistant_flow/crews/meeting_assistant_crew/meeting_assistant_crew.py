from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI

from meeting_assistant_flow.types import (
    MeetingTaskList,
)
from obiguard import OBIGUARD_GATEWAY_URL


@CrewBase
class MeetingAssistantCrew:
    """Meeting Assistant Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = ChatOpenAI(model="gpt-4")

    def __init__(self, obiguard_client):
        self.obiguard_client = obiguard_client
        self.llm = LLM(
            model="openai/Qwen/Qwen2.5-32B-Instruct",
            base_url=OBIGUARD_GATEWAY_URL,
            api_key='N/A',
            extra_headers=self.obiguard_client.copy_headers()
        )

    @agent
    def meeting_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_analyzer"],
            llm=self.llm,
        )

    @task
    def analyze_meeting(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_meeting"],
            output_pydantic=MeetingTaskList,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Meeting Issue Generation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
