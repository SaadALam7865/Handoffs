import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent,Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig

load_dotenv(find_dotenv())
gemini_api_key=os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError('GEMINI_API_KEY is not set in the env..')

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai'
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model='gemini-2.0-flash'
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name='Assiatant',
    instructions='You are a helpful Assiatant and always Response Humbly and always Says calling Name the Saad',
    model=model
)

res = Runner.run_sync(
    starting_agent=agent,
    input='what is handoffs in agentic ai',
    run_config=config
)

print(res.final_output)
