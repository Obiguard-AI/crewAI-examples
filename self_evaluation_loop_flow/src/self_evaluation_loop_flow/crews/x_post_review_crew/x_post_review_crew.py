from typing import Optional

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from obiguard import Obiguard, OBIGUARD_GATEWAY_URL
from pydantic import BaseModel
from self_evaluation_loop_flow.tools.CharacterCounterTool import CharacterCounterTool


class XPostVerification(BaseModel):
    valid: bool
    feedback: Optional[str]


@CrewBase
class XPostReviewCrew:
    """X Post Review Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, obiguard_client: Obiguard):
        self.obiguard_client = obiguard_client

    @agent
    def x_post_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config["x_post_verifier"],
            tools=[CharacterCounterTool()],
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    @task
    def verify_x_post(self) -> Task:
        return Task(
            config=self.tasks_config["verify_x_post"],
            output_pydantic=XPostVerification,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the X Post Review Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
