def build_prompt(
    history: list[tuple[str, str]],
    user_input: str,
    emotion_label: str = None,
    model_type: str = "gpt",
    tokenizer = None
) -> str:
    """
    Build better prompts with proper formatting for each model type
    """
    
    if model_type == "dialogpt" and tokenizer:
        # DialoGPT expects alternating turns separated by EOS tokens
        # Format: Human message<EOS>Bot response<EOS>Human message<EOS>
        
        conversation = ""
        
        # Add recent history (max 3 exchanges to prevent confusion)
        for user_msg, bot_msg in history[-2:]:  # Reduced from 3 to 2
            conversation += f"{user_msg.strip()}{tokenizer.eos_token}{bot_msg.strip()}{tokenizer.eos_token}"
        
        # Add current user input
        conversation += f"{user_input.strip()}{tokenizer.eos_token}"
        
        return conversation
    
    else:
        # For GPT-2 based models, use clear instruction format
        lines = [
            "You are a helpful and empathetic AI assistant named Soul Sync.",
            "Respond naturally and helpfully to the human's questions and comments.",
            "Keep responses brief, relevant, and conversational.",
            ""
        ]
        
        # Add emotion context if provided
        if emotion_label and emotion_label != "neutral":
            emotion_prompts = {
                "happy": "The user seems happy. Share in their positive mood.",
                "excited": "The user seems excited. Match their enthusiasm appropriately.",
                "sad": "The user seems sad. Be supportive and understanding.",
                "frustrated": "The user seems frustrated. Be patient and helpful.",
                "confused": "The user seems confused. Provide clear, simple explanations.",
                "angry": "The user seems angry. Stay calm and be understanding."
            }
            lines.append(emotion_prompts.get(emotion_label, f"The user seems {emotion_label}. Respond appropriately."))
            lines.append("")
        
        # Add recent conversation history
        for user_msg, bot_msg in history[-2:]:
            lines.append(f"Human: {user_msg.strip()}")
            lines.append(f"Soul Sync: {bot_msg.strip()}")
        
        # Add current exchange
        lines.append(f"Human: {user_input.strip()}")
        lines.append("Soul Sync:")
        
        return "\n".join(lines)


def truncate_history(
    history: list[tuple[str, str]],
    tokenizer,
    max_tokens: int = 200  # Reduced for better performance
) -> list[tuple[str, str]]:
    """
    Smart history truncation to maintain context while preventing overload
    """
    if not history:
        return history
    
    # Start with most recent exchanges
    truncated = []
    total_tokens = 0
    
    # Work backwards through history
    for user_msg, bot_msg in reversed(history):
        # Rough token estimation (more accurate)
        exchange_tokens = len(tokenizer.encode(f"{user_msg} {bot_msg}"))
        
        if total_tokens + exchange_tokens > max_tokens:
            break
            
        truncated.insert(0, (user_msg, bot_msg))
        total_tokens += exchange_tokens
        
        # Limit to max 3 exchanges regardless of token count
        if len(truncated) >= 3:
            break
    
    return truncated


def validate_response(response: str, user_input: str) -> tuple[bool, str]:
    """
    Validate if the response makes sense and provide fallback if needed
    """
    if not response or len(response.strip()) < 2:
        return False, "I'm not sure how to respond to that. Could you rephrase?"
    
    # Check for common nonsensical patterns
    nonsense_indicators = [
        len(response) > 200,  # Too long
        response.count('.') > 5,  # Too many sentences
        'hahaha' in response.lower() and 'what' in user_input.lower(),  # Inappropriate laughter
        response.lower().startswith(('um', 'uh', 'umm')) and len(response) < 20,  # Filler words only
        'this is me trying to find out' in response.lower(),  # Generic confusion
        response.count('thank you') > 2,  # Excessive politeness
    ]
    
    if any(nonsense_indicators):
        # Provide contextual fallback based on user input
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['what', 'capital', 'who', 'when', 'where', 'how']):
            fallback = "I'd be happy to help answer your question. Could you be more specific about what you're looking for?"
        elif any(word in user_lower for word in ['hi', 'hello', 'hey']):
            fallback = "Hello! I'm Soul Sync. How are you doing today?"
        elif any(word in user_lower for word in ['help', 'need', 'can you']):
            fallback = "I'm here to help! What do you need assistance with?"
        elif any(word in user_lower for word in ['feel', 'feeling', 'emotion']):
            fallback = "I understand you're sharing how you feel. Tell me more about what's on your mind."
        else:
            fallback = "That's interesting. Tell me more about that."
            
        return False, fallback
    
    return True, response


def get_fallback_response(user_input: str, emotion_label: str = "neutral") -> str:
    """
    Generate contextual fallback responses when model fails
    """
    user_lower = user_input.lower()
    
    # Question patterns
    if any(word in user_lower for word in ['what', 'capital', 'who', 'when', 'where']):
        if 'capital' in user_lower and 'india' in user_lower:
            return "The capital of India is New Delhi."
        return "That's a great question! I'd be happy to help you find that information."
    
    # Greeting patterns
    if any(word in user_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        greetings = [
            "Hello! How are you doing today?",
            "Hi there! What's on your mind?",
            "Hey! How can I help you today?",
            "Hello! I'm here if you need someone to talk to."
        ]
        import random
        return random.choice(greetings)
    
    # Grammar/language questions
    if any(word in user_lower for word in ['past form', 'present form', 'grammar']):
        if 'past form' in user_lower and 'do' in user_lower:
            return "The past form of 'do' is 'did'."
        return "I can help with grammar questions! What would you like to know?"
    
    # Emotional responses based on detected emotion
    emotion_responses = {
        "happy": "I'm glad you're feeling positive! What's making you happy today?",
        "sad": "I'm sorry you're feeling down. Would you like to talk about what's bothering you?",
        "frustrated": "I can sense your frustration. How can I help make things better?",
        "confused": "It's okay to feel confused sometimes. What's puzzling you?",
        "angry": "I understand you might be upset. Take your time, and let me know how I can help.",
        "excited": "Your excitement is wonderful! What's got you so energized?"
    }
    
    if emotion_label in emotion_responses:
        return emotion_responses[emotion_label]
    
    # Default responses
    defaults = [
        "I'm listening. What would you like to talk about?",
        "That's interesting. Tell me more.",
        "I'm here to help. What's on your mind?",
        "I understand. How can I assist you today?"
    ]
    
    import random
    return random.choice(defaults)