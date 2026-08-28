import os

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

client = Anthropic(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL"),
)

stream = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-opus-5",
    stream=True,

)
for event in stream:
    print(event)