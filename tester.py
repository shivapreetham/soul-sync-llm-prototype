import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_plain_model(model_name="distilgpt2"):
    """
    Load a plain causal LM (DistilGPT2) and generate a short continuation
    with a conversational prefix to reduce echoing/weird tokens.
    """
    print(f"\n=== Testing plain model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Make pad_token same as eos_token to avoid attention-mask warning
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    # Example prompt with explicit conversational context
    prompt = "User: Hello, how are you today?\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    max_new_tokens = 30

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - t0

    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        output_ids[0, input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    print(f"Prompt: {prompt!r}")
    print(f"Generated continuation: {generated!r}")
    print(f"Inference time: {elapsed:.2f}s")


def test_chat_model(model_name="microsoft/DialoGPT-small"):
    """
    Load DialoGPT-small and do a 2-turn exchange, with conversational context prefix.
    Maintains chat_history_ids for context.
    """
    print(f"\n=== Testing chat model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    # Simple two-turn conversation, but use explicit "User:" / "Bot:" tags
    chat_history_ids = None

    for step, user_input in enumerate(["Hi there!", "How's the weather?"]):
        # Build full input with tags
        # If first turn, start fresh
        if chat_history_ids is None:
            prompt = f"User: {user_input}\nBot:"
        else:
            # Decode previous history to text, append new turn
            prev = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
            # prev contains everything from start; ensure it ends with Bot: response
            prompt = prev + f"\nUser: {user_input}\nBot:"

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        t0 = time.time()
        with torch.no_grad():
            chat_history_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        # Extract generated part: everything after the prompt
        # We can decode full output, then strip off the prompt text
        full_text = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
        # The response is the part after the last "\nBot:"
        if "\nBot:" in full_text:
            response = full_text.split("\nBot:")[-1].strip()
        else:
            response = full_text.strip()

        print(f"\nUser ({step+1}): {user_input}")
        print(f"Bot  : {response}")
        print(f"Inference time: {elapsed:.2f}s")


if __name__ == "__main__":
    # Run both tests
    test_plain_model("distilgpt2")
    test_chat_model("microsoft/DialoGPT-small")
