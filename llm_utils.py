import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer_and_model(model_name: str, device: torch.device = None):
    """
    Load tokenizer and model, set pad_token, move model to device.
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
    return tokenizer, model, device

def generate_response(prompt: str,
                      tokenizer,
                      model,
                      device: torch.device,
                      max_new_tokens: int = 50,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      do_sample: bool = True,
                      warmup: bool = True):
    """
    Generate text given prompt. Returns (text, elapsed_seconds).
    """
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if warmup:
        try:
            _ = model.generate(input_ids, attention_mask=attention_mask,
                               max_new_tokens=10,
                               pad_token_id=model.config.pad_token_id)
        except Exception:
            pass

    start = time.time()
    out_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=model.config.pad_token_id
    )
    elapsed = time.time() - start
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text, elapsed
