from crewai import Agent, LLM
from textwrap import dedent
from obiguard import Obiguard, OBIGUARD_GATEWAY_URL


# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class CustomAgents:
    def __init__(self, obiguard_client: Obiguard):
        self.obiguard_client = obiguard_client

    def agent_1_name(self):
        return Agent(
            role="Define agent 1 role here",
            backstory=dedent(f"""Define agent 1 backstory here"""),
            goal=dedent(f"""Define agent 1 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )

    def agent_2_name(self):
        return Agent(
            role="Define agent 2 role here",
            backstory=dedent(f"""Define agent 2 backstory here"""),
            goal=dedent(f"""Define agent 2 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
        )
