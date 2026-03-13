# Request
from openai import OpenAI
client = OpenAI(
    base_url="https://guest-api.sktax.chat/v1",
    api_key="sktax-XyeKFrq67ZjS4EpsDlrHHXV8it" #A.X 공개키
)

completion = client.chat.completions.create(
    model="skt/A.X-3.1-Light",
    messages=[
    {"role": "user", "content": ""}
    ]
)
print(completion.choices[0].message.content)