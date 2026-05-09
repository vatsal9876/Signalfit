import os
import json
from dotenv import load_dotenv
from groq import Groq
from prompts import state_system_prompt

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_state(messages):
    conversation=""

    for msg in messages:
        conversation+=f"{msg['role']}: {msg['content']}\n"

    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": state_system_prompt
            },
            {
                "role": "user",
                "content": conversation
            }
        ],
        temperature=0
    )

    content=response.choices[0].message.content
    return json.loads(content)

if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": (
                "Hiring a mid-level Java "
                "backend engineer with "
                "stakeholder communication"
            )
        }
    ]

    state = extract_state(messages)

    print(json.dumps(state, indent=2))
