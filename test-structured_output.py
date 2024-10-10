from pydantic import BaseModel
from openai import OpenAI

# client = OpenAI(base_url="http://localhost:8000/v1", api_key="")


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# completion = client.beta.chat.completions.parse(
#     model="llama3.2:1b",
#     messages=[
#         {"role": "system", "content": "Extract the event information."},
#         {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
#     ],
#     # response_format=CalendarEvent,
# )

# event = completion.choices[0].message.parsed

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ollama",  # required, but unused
)

response = client.beta.chat.completions.parse(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ],
    response_format=CalendarEvent,
)
print(response.choices[0].message.parsed)
