import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv(find_dotenv())
bank_api_key = os.getenv('GEMINI_API_KEY')

if not bank_api_key:
    raise ValueError("BANK_API_KEY not set in .env file")

# Setup external client
external_client = AsyncOpenAI(
    api_key=bank_api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai'
)

# Define model
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model='gemini-2.0-flash'
)

# Define config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# --------------------------
# Specialized Agents
# --------------------------

card_agent = Agent(
    name='Card Support Agent',
    instructions='You handle credit and debit card issue. Help users with lost/stolen cards, new cards request and activation.'

)
loan_agent = Agent(
    name='Loan Support Agent',
    instructions='You manage loan-related queries. Help with loan status, applications, and EMI info.'
)
fraud_agent = Agent(
    name='Fraud Detection Agent',
    instructions='You specialize in handling fraud reports,suspicious activities,and freezing accounts.'
)
# --------------------------
# Main General Agent
# --------------------------

main_agent = Agent(
    name='Bank Customer Support Agent',
    instructions="You're the general support agent for a bank. Handle basic  questions or hand off to card, loan, or fraud agents as needed.",
    handoffs=[card_agent,loan_agent,fraud_agent]
)
# --------------------------
# Run the agent with input
#
response = Runner.run_sync(
    main_agent,
    input='I want to apply for a new credit card.',
    run_config=config
)

# --------------------------
# Output Result
# --------------------------
print(f"\n🧾 Final Agent Response:")
print(response.final_output)

