from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Runconfig
from dotenv import load_dotenv
import os


load_dotenv()

# Load Gemini API key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define provider and model
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Define run config
config = Runconfig(
    model=model,
    model_provider=provider,
    tracking_disabled=True,
)

# Tool agents
capital_agent = Agent(
    name="CapitalFinder",
    instructions="Return the capital city of the country mentioned in the message.",
    model=model
)

language_agent = Agent(
    name="LanguageFinder",
    instructions="Return the main language(s) spoken in the country mentioned.",
    model=model
)

population_agent = Agent(
    name="PopulationFinder",
    instructions="Return the population of the country mentioned.",
    model=model
)

orchestrator = Agent(
    name="CountryOrchestrator",
    instructions="You are a country information assistant. Just extract the country name from the user message.",
    model=model
)

# Input
user_input = input("Ask me about any country (capital, language, population): ")

# Run
try:
    # Step 1: Extract country
    country_result = Runner.run(orchestrator, user_input, runconfig=config)
    country = country_result.final_output.strip()

    # Step 2: Get details
    cap_result = Runner.run(capital_agent, country, runconfig=config).final_output
    lang_result = Runner.run(language_agent, country, runconfig=config).final_output
    pop_result = Runner.run(population_agent, country, runconfig=config).final_output

    # Output
    print(f"\n📍 Country: {country}")
    print(f"🏛️ Capital: {cap_result}")
    print(f"🗣️ Language(s): {lang_result}")
    print(f"👥 Population: {pop_result}")

except Exception as e:
    print(f"❌ Error: {str(e)}")