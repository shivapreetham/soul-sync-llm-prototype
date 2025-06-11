def build_prompt(
    history: list[tuple[str, str]],
    user_input: str,
    emotion_label: str = None,
    model_type: str = "gpt",
    tokenizer = None
) -> str:
    """
    Build better prompts based on model type to reduce repetition
    """
    
    if model_type == "dialogpt" and tokenizer:
        # DialoGPT works better with this format
        lines = []
        
        # Add emotion context if provided
        if emotion_label and emotion_label != "neutral":
            lines.append(f"The human seems {emotion_label}. Respond empathetically.")
        
        # Add conversation history in DialoGPT format
        for user_msg, bot_msg in history[-3:]:  # Only last 3 turns
            lines.append(f"{user_msg}{tokenizer.eos_token}{bot_msg}{tokenizer.eos_token}")
        
        # Add current input
        lines.append(f"{user_input}{tokenizer.eos_token}")
        
        return "".join(lines)
    
    else:
        # For GPT-2 based models, use instruction format
        lines = [
            "You are Soul Sync, an empathetic AI assistant. Have a natural conversation.",
            ""
        ]
        
        if emotion_label and emotion_label != "neutral":
            lines.append(f"The user appears to be feeling {emotion_label}. Respond appropriately.")
            lines.append("")
        
        # Add recent history
        for user_msg, bot_msg in history[-2:]:  # Only last 2 turns
            lines.append(f"Human: {user_msg}")
            lines.append(f"Assistant: {bot_msg}")
        
        lines.append(f"Human: {user_input}")
        lines.append("Assistant:")
        
        return "\n".join(lines)

def truncate_history(
    history: list[tuple[str, str]],
    tokenizer,
    max_tokens: int = 300
) -> list[tuple[str, str]]:
    """
    More aggressive truncation to prevent repetition
    """
    if not history:
        return history
    
    # Keep only recent history
    h = history[-4:]  # Max 4 recent exchanges
    
    while h:
        # Estimate token count (rough approximation)
        total_text = " ".join([f"{u} {b}" for u, b in h])
        estimated_tokens = len(total_text.split()) * 1.3  # Rough token estimate
        
        if estimated_tokens <= max_tokens:
            break
        h.pop(0)
    
    return h

