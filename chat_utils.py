# chat_utils.py
from transformers import AutoTokenizer

def build_chat_prompt(history: list[tuple[str, str]], user_input: str):
    """
    history: list of (user_msg, bot_msg) tuples in order.
    Returns a single prompt string:
      User: ... 
      Assistant: ...
      ...
      User: {user_input}
      Assistant:
    """
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def truncate_history(history: list[tuple[str, str]], tokenizer: AutoTokenizer, max_tokens: int = 512):
    """
    Truncate oldest turns so that tokenized prompt length stays <= max_tokens.
    Returns truncated history.
    """
    # Build prompt tokens length; remove oldest until within limit
    while history:
        prompt = build_chat_prompt(history, "")
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if tokens.shape[0] <= max_tokens:
            break
        # drop oldest turn
        history.pop(0)
    return history
