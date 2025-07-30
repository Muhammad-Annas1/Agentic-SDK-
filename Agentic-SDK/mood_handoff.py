from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Runconfig 
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini Model setup
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

config = Runconfig(
    model=model,
    model_provider=provider,
    tracking_disabled=True
)

# Step 1: Mood Checker Agent
mood_checker = Agent(
    name="MoodChecker",
    instructions="Check user's message and identify their mood using one word like: happy, sad, excited, angry, or stressed.",
    model=model
)

# Step 2: Mood Suggester Agent
mood_suggester = Agent(
    name="MoodSuggester",
    instructions="Suggest a short, comforting activity if the mood is sad or stressed. Be kind and supportive.",
    model=model
)

# Input from user
user_input = input("💬 How are you feeling today? ")

# Run MoodChecker
mood_result = Runner.run(mood_checker, user_input, runconfig=config)
mood = mood_result.final_output.strip().lower()

print(f"\n🧠 Detected Mood: {mood}")

# Agent Handoff Logic
if mood in ["sad", "stressed"]:
    # Handoff to MoodSuggester
    suggestion_result = mood_checker.handoff(
        mood_suggester,
        input=f"My mood is {mood}. Please suggest something helpful.",
        runconfig=config
    )
    print(f"💡 Suggested Activity: {suggestion_result.final_output.strip()}")
else:
    print("😊 You seem to be doing fine! Keep it up!")