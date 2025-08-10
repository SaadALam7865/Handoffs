import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner,handoff, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
 
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
custom_card_handoff = handoff(
    agent=card_agent,
    tool_name_override='custom_card_tool',
    tool_description_override='handels credit and debit card issues with all problems realted to cards...'
)

loan_agent = Agent(
    name='Loan Support Agent',
    instructions='You manage loan-related queries. Help with loan status, applications, and EMI info.'
)
fraud_agent = Agent(
    name='Fraud Detection Agent',
    instructions='You specialize in handling fraud reports,suspicious activities,and freezing accounts.'
)

custom_load_handoff = handoff(
    agent=loan_agent,
    tool_name_override='custom_loan_tool',
    tool_description_override='Handles User loan realted quesries with extra care. and solve the user problems.'
)

custom_fraud_handoff = handoff(
    agent=fraud_agent,
    tool_name_override='custom_fraud_tool',
    tool_description_override='Hanndles fraud realted queries of the user with extra care..'
)
# --------------------------
# Main General Agent
# --------------------------

main_agent = Agent(
    name='Bank Customer Support Agent',
    instructions="You're the general support agent for a bank. Handle basic  questions or hand off to card, loan, or fraud agents as needed.",
    handoffs=[custom_card_handoff,custom_load_handoff,custom_fraud_handoff],
    
    
)
# --------------------------
# Run the agent with input
#
response = Runner.run_sync(
    main_agent,
    input='i wan to make new credit card bcz my card is stolen in yesterday?',
    run_config=config
)

# --------------------------
# Output Result
# --------------------------
print(f"\n🧾 Final Agent Response:")
print(response.final_output)

