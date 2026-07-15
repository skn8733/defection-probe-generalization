from together import Together

client = Together()

# Test 1: confirm connection works
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Reference",
    messages=[{"role": "user", "content": "Hello, respond in one sentence."}],
    max_tokens=50,
)
print("Response:", response.choices[0].message.content)