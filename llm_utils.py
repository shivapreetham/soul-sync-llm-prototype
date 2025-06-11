
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def load_model(model_name="microsoft/DialoGPT-medium", device=None):
    """
    Load model with better configuration
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return tokenizer, model, device

def clean_response(text: str, context_words: list) -> str:
    """
    Clean up the generated response to remove repetition and artifacts
    """
    if not text:
        return ""
    
    # Remove common repetitive patterns
    text = re.sub(r'(.{10,}?)\1{2,}', r'\1', text)  # Remove repetitions of 10+ chars
    
    # Remove artifacts and stop tokens
    artifacts = ["Human:", "Assistant:", "User:", "<|", "|>", "�", "\n\nHuman", "\n\nUser"]
    for artifact in artifacts:
        text = text.split(artifact)[0]  # Take everything before the artifact
    
    # Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.,!?;:]+', '', text).strip()
    
    # Remove incomplete sentences at the end
    if text.endswith(('...', '..', '.')):
        pass  # Keep intentional ellipses
    elif text and not text[-1] in '.!?':
        # If doesn't end with punctuation, find last complete sentence
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1 and sentences[-1].strip():
            # Remove incomplete last sentence
            text = '.'.join(sentences[:-1])
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
    
    # Limit length to prevent rambling
    if len(text) > 150:
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            text = sentences[0] + '.'
        else:
            text = text[:150].rsplit(' ', 1)[0] + "..."
    
    return text.strip()

def generate_response(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_p: float = 0.85,
    top_k: int = 50,
    do_sample: bool = True,
    history=None
) -> str:
    """
    Enhanced generation with proper attention masks and fixed-size handling
    """
    
    # Tokenize with proper padding and truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True,
        return_attention_mask=True
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Get sequence length for proper generation bounds
    seq_len = input_ids.shape[-1]
    max_length = min(seq_len + max_new_tokens, 512)  # Don't exceed model's max length
    
    # Create bad_words list from recent history to prevent repetition
    bad_words_ids = []
    if history and len(history) > 0:
        # Get last few bot responses to avoid repeating
        recent_responses = [resp for _, resp in history[-2:]]
        for resp in recent_responses:
            words = resp.split()[:3]  # First 3 words of recent responses
            for word in words:
                if len(word) > 3:  # Only longer words
                    try:
                        word_ids = tokenizer.encode(word, add_special_tokens=False)
                        if word_ids and len(word_ids) == 1:  # Single token words only
                            bad_words_ids.append(word_ids)
                    except:
                        continue
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=seq_len + 5,  # Ensure minimum response length
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,  # Moderate repetition penalty
            no_repeat_ngram_size=3,   # Prevent 3-gram repetition
            bad_words_ids=bad_words_ids[:5] if bad_words_ids else None,  # Limit bad words
            early_stopping=True,
            use_cache=True
        )
    
    # Decode only the new tokens
    new_tokens = output[0][seq_len:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean the response
    cleaned_response = clean_response(new_text, prompt.split()[-10:])  # Last 10 words as context
    
    # Fallback responses if generation fails
    if not cleaned_response or len(cleaned_response.strip()) < 3:
        fallback_responses = [
            "I understand. Tell me more about that.",
            "That's interesting. How do you feel about it?",
            "I'm listening. What else would you like to share?",
            "Thank you for sharing. What's on your mind?",
            "I see. How can I help you with that?"
        ]
        import random
        cleaned_response = random.choice(fallback_responses)
    
    return cleaned_response.strip()
