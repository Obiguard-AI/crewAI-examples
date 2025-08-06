from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from obiguard import Obiguard, OBIGUARD_GATEWAY_URL
from self_evaluation_loop_flow.tools.CharacterCounterTool import CharacterCounterTool


@CrewBase
class ShakespeareanXPostCrew:
    """Shakespearean X Post Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, obiguard_client: Obiguard):
        self.obiguard_client = obiguard_client

    @agent
    def shakespearean_bard(self) -> Agent:
        return Agent(
            config=self.agents_config["shakespearean_bard"],
            tools=[CharacterCounterTool()],
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    @task
    def write_x_post(self) -> Task:
        return Task(
            config=self.tasks_config["write_x_post"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Shakespearean X Post Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
