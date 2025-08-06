from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from obiguard import Obiguard, OBIGUARD_GATEWAY_URL

# Knowledge sources
pdf_source = PDFKnowledgeSource(
    file_paths=["meta_quest_manual.pdf"]
)

@CrewBase
class MetaQuestKnowledge():
	"""MetaQuestKnowledge crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, obiguard_client: Obiguard):
		self.obiguard_client = obiguard_client

	@agent
	def meta_quest_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['meta_quest_expert'],
			verbose=True,
            llm=LLM(
                model="openai/Qwen/Qwen2.5-32B-Instruct",
                base_url=OBIGUARD_GATEWAY_URL,
                api_key='N/A',
                extra_headers=self.obiguard_client.copy_headers()
            )
		)

	@task
	def answer_question_task(self) -> Task:
		return Task(
			config=self.tasks_config['answer_question_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MetaQuestKnowledge crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			knowledge_sources=[
				pdf_source
			]
		)
