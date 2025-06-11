import torch, time, psutil, os
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    return tokenizer, model

def profile(model_name, prompt="Hello world", max_new_tokens=40):
    device = torch.device("cpu")
    print(f"\nProfiling {model_name}")
    tokenizer, model = load_model(model_name, device)
    # Warm-up
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        _ = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=5)
    # Time
    start = time.time()
    with torch.no_grad():
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens)
    elapsed = time.time() - start
    # Memory
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss / (1024**2)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Time: {elapsed:.2f} s; RAM: {mem:.1f} MB; Sample: {text[:80].replace(chr(10),' ')}")
    del model, tokenizer
    torch.cuda.empty_cache()

if __name__=="__main__":
    for m in ["EleutherAI/gpt-neo-125M", "distilgpt2", "gpt2"]:
        profile(m)
