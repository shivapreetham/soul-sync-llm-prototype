from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

def load_model(model_name: str, device: torch.device = None):
    """
    Load tokenizer and model, move model to device, set pad token.
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if device is None:
        device = torch.device("cpu")
    model.to(device)
    return tokenizer, model

def generate_response(prompt: str,
                      tokenizer,
                      model,
                      device: torch.device,
                      max_new_tokens: int = 30,
                      warmup: bool = True):
    """
    Generate text for a prompt, returning (generated_text, elapsed_seconds).
    Uses attention_mask and pad_token for safety.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    start = time.time()
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.pad_token_id
    )
    elapsed = time.time() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, elapsed

def main():
    model_name = "EleutherAI/gpt-neo-125M"  # @param ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-6.7B"]
    device = torch.device("cpu")

    tokenizer, model = load_model(model_name, device)

    # Example prompt
    prompt = "Hello! Today I feel curious because"
    generated, elapsed = generate_response(prompt, tokenizer, model, device, max_new_tokens=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated ({elapsed:.2f} s): {generated}")

if __name__ == "__main__":
    main()
