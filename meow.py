import openai

client = openai.OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)