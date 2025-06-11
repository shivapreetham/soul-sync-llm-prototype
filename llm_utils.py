import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random

def load_model(model_name="microsoft/DialoGPT-medium", device=None):
    """
    Load model with optimized configuration
    """
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Configure tokenizer properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return tokenizer, model, device


def clean_response(text: str, user_input: str = "") -> str:
    """
    Enhanced response cleaning with better context awareness
    """
    if not text:
        return ""
    
    original_text = text
    
    # Remove common artifacts first
    artifacts = [
        "Human:", "Assistant:", "User:", "Soul Sync:", "Bot:",
        "<|", "|>", "�", "\n\nHuman", "\n\nUser", "\n\nAssistant",
        AutoTokenizer.eos_token if 'tokenizer' in globals() else "<|endoftext|>"
    ]
    
    for artifact in artifacts:
        if artifact in text:
            text = text.split(artifact)[0]
    
    # Remove repetitive patterns (be more aggressive)
    text = re.sub(r'(.{5,}?)\1{2,}', r'\1', text)  # Remove repetitions of 5+ chars
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove word repetitions
    
    # Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.,!?;:]+', '', text).strip()
    text = re.sub(r'[.,!?;:]{2,}', '.', text)  # Multiple punctuation to single
    
    # Remove filler words at the start
    text = re.sub(r'^(um+|uh+|er+|ah+|well|so|like)\s+', '', text, flags=re.IGNORECASE)
    
    # Handle incomplete sentences more intelligently
    if text and not text[-1] in '.!?':
        # If it ends mid-sentence and we have complete sentences, truncate
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1 and len(sentences[0].strip()) > 10:
            text = sentences[0].strip() + '.'
        elif len(text) > 80:  # Long incomplete response
            text = text[:80].rsplit(' ', 1)[0] + "..."
    
    # Final cleanup
    text = text.strip()
    
    # Quality checks
    if len(text) < 3:
        return ""
    
    # Check for nonsensical patterns
    if (text.count('haha') > 2 or 
        text.lower().startswith(('umm...', 'uh...', 'what?')) or
        'this is me trying to find out what you want' in text.lower()):
        return ""
    
    return text


def generate_response(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 30,  # Reduced for better quality
    temperature: float = 0.7,  # Lowered for more coherent responses
    top_p: float = 0.8,       # Lowered for better focus
    top_k: int = 40,          # Lowered for better quality
    do_sample: bool = True,
    history=None,
    user_input: str = ""
) -> str:
    """
    Enhanced generation with better parameter tuning and fallback handling
    """
    
    try:
        # Tokenize with proper handling
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400,  # Reduced to prevent context confusion
            padding=True,
            return_attention_mask=True
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Calculate generation bounds
        seq_len = input_ids.shape[-1]
        max_length = min(seq_len + max_new_tokens, 450)
        
        # Generate bad words from repetitive history
        bad_words_ids = []
        if history and len(history) > 1:
            # Get recent bot responses to avoid repetition
            recent_responses = [resp for _, resp in history[-2:]]
            for resp in recent_responses:
                # Tokenize common phrases from recent responses
                words = resp.split()[:2]  # First 2 words only
                for word in words:
                    if len(word) > 3:
                        try:
                            word_ids = tokenizer.encode(word, add_special_tokens=False)
                            if word_ids and len(word_ids) == 1:
                                bad_words_ids.append(word_ids)
                        except:
                            continue
        
        # Generation with optimized parameters
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=seq_len + 3,  # Minimum response
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,   # Increased to reduce repetition
                no_repeat_ngram_size=2,   # Prevent 2-gram repetition
                bad_words_ids=bad_words_ids[:3] if bad_words_ids else None,
                early_stopping=True,
                use_cache=True,
                num_beams=1,  # Greedy-like for consistency
                length_penalty=0.8  # Slight penalty for length
            )
        
        # Decode only new tokens
        new_tokens = output[0][seq_len:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean the response
        cleaned_response = clean_response(new_text, user_input)
        
        return cleaned_response
        
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


def get_smart_fallback(user_input: str, emotion_label: str = "neutral") -> str:
    """
    Provide intelligent fallback responses for common queries
    """
    user_lower = user_input.lower().strip()
    
    # Direct factual questions
    factual_responses = {
        "what is capital of india": "The capital of India is New Delhi.",
        "what is the capital of india": "The capital of India is New Delhi.",
        "capital of india": "The capital of India is New Delhi.",
        "past form of do": "The past form of 'do' is 'did'.",
        "present form of do": "Actually, 'do' is the present form. The past form is 'did'.",
        "what is 2+2": "2 + 2 equals 4.",
        "who is the president of usa": "As of my last update, the current president information may have changed. Could you check recent news for the most current information?"
    }
    
    # Check for exact matches first
    if user_lower in factual_responses:
        return factual_responses[user_lower]
    
    # Partial matches for flexibility
    if "capital" in user_lower and "india" in user_lower:
        return "The capital of India is New Delhi."
    
    if "past form" in user_lower and "do" in user_lower:
        return "The past form of 'do' is 'did'."
    
    if "present form" in user_lower and ("do" in user_lower or "did" in user_lower):
        return "The present form is 'do' and the past form is 'did'."
    
    # Greeting responses
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(greeting in user_lower for greeting in greetings):
        responses = [
            "Hello! How are you today?",
            "Hi there! What can I help you with?",
            "Hey! How are you feeling today?",
            "Hello! I'm here to chat with you."
        ]
        return random.choice(responses)
    
    # Emotion-based responses
    if emotion_label != "neutral":
        emotion_responses = {
            "happy": "That's wonderful! I'm glad you're feeling good. What's making you happy?",
            "sad": "I'm sorry you're feeling down. Would you like to talk about what's bothering you?",
            "frustrated": "I understand you're feeling frustrated. How can I help?",
            "confused": "It's okay to feel confused. What would you like me to clarify?",
            "angry": "I can sense you're upset. I'm here to listen if you need to talk.",
            "excited": "Your excitement is contagious! What's got you so energized?"
        }
        if emotion_label in emotion_responses:
            return emotion_responses[emotion_label]
    
    # Default helpful responses
    default_responses = [
        "I'm here to help. What would you like to know?",
        "That's interesting. Tell me more about that.",
        "I'm listening. What's on your mind?",
        "How can I assist you today?",
        "I'd be happy to help. Could you tell me more?"
    ]
    
    return random.choice(default_responses)