from huggingface_hub import InferenceClient

MODEL_URL = 'http://localhost:8000'

model_client = InferenceClient(model=MODEL_URL)

prompt

response = model_client.text_generation(prompt=prompt, max_new_tokens=2048)

