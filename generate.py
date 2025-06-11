from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "EleutherAI/gpt-neo-125M"

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure model uses CPU
    device = torch.device("cpu")
    model.to(device)

    prompt = "Hello! Today I feel curious because"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate up to 30 new tokens
    print("Generating...")
    output_ids = model.generate(input_ids, max_new_tokens=30)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Generated Text ===")
    print(text)

if __name__ == "__main__":
    main()
