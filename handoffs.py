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

billing_agent = Agent(
    name='Billing Agent',
    instructions='You are a Billing Agent yor task to dealing billing issue of the user',
    
)

refund_agent = Agent(
    name='Refund Agent',
    instructions='you are a refund agent your task is to handling refund the process'
)

main_agen = Agent(
    name='Customer Support Agent',
    instructions='You are a helpful customer support Agent.',
    handoffs=[billing_agent,refund_agent]
)

res = Runner.run_sync(
    main_agen,
    input='i wan to make new credit card bcz my card is stolen in yesterday?',
    run_config=config
)

print(res.final_output)