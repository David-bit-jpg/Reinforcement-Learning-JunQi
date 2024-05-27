import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def test_dialogue():
    print("=== Chat with the model ===")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit', 'q']:
            break
        response = generate_text(prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    test_dialogue()
