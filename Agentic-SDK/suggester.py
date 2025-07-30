from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Runconfig # type: ignore
from dotenv import load_dotenv  

import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

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
    tracking_disabled=True,
)

agent = Agent(
    name="Smart Store Agent",
    instructions=(
        "You are a smart and friendly store assistant. "
        "When a user describes a health issue (e.g., headache, fever, flu, cough, body pain), "
        "you must suggest a suitable over-the-counter (OTC) product or remedy. "
        "Always explain *why* that product is helpful for the mentioned symptom. "
        "For example:\n"
        "- If someone says 'I have a headache', suggest Panadol or Ibuprofen and explain it relieves pain.\n"
        "- For flu, suggest flu tablets or a decongestant with reason.\n"
        "- For cough, suggest cough syrup based on dry or wet cough.\n"
        "Avoid giving serious medical advice. Always recommend the user consult a doctor if symptoms are severe.\n"
        "If you don't know the answer, simply respond: 'I'm not sure, please consult a pharmacist or doctor.'"
    ),
    model=model,
)

user_input = input("What do you need help with?(e.g, I have a cough ): ")

result = Runner.run(                    
    agent,
    user_input,
    runconfig=config
)   

print(result.final_output)  # Output the final response from the agent