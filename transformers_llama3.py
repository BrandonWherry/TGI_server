from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup
model_id = "casperhansen/llama-3-8b-instruct-awq"
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.to(device)

# Example chat messages
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

# Apply the chat prompt method from the tokenizer
prompt = tokenizer.apply_chat_prompt(messages, tokenize=False, add_generation_prompt=True)

# Prepare input for generation
inputs = tokenizer(prompt, return_tensors="pt").to(device)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Generate response using model
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

# Extract and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
