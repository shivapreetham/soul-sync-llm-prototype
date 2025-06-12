import streamlit as st
import torch
from llm_utils import load_model, generate_response, get_smart_fallback
from chat_utils import build_prompt, truncate_history, validate_response, detect_conversation_quality

st.set_page_config(
    page_title="Soul Sync Chat", 
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_model(model_name: str):
    """Initialize model with caching"""
    return load_model(model_name)

def detect_emotion_from_text(text: str) -> str:
    """
    Simple emotion detection based on keywords and patterns
    """
    text_lower = text.lower()
    
    # Emotion keyword mapping
    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "great", "awesome", "wonderful", "amazing", "love", "fantastic"],
        "sad": ["sad", "depressed", "down", "upset", "hurt", "disappointed", "cry", "awful", "terrible"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated", "hate", "frustrated", "pissed"],
        "frustrated": ["frustrated", "stuck", "annoying", "difficult", "struggling", "can't", "won't work"],
        "confused": ["confused", "don't understand", "unclear", "puzzled", "lost", "what", "how", "why"],
        "excited": ["excited", "can't wait", "amazing", "incredible", "wow", "omg", "awesome", "fantastic"],
        "anxious": ["worried", "nervous", "anxious", "scared", "afraid", "concerned", "stress", "panic"],
        "tired": ["tired", "exhausted", "sleepy", "worn out", "drained", "fatigue"]
    }
    
    # Punctuation patterns
    if text.count('!') >= 2:
        return "excited"
    elif text.count('?') >= 2:
        return "confused"
    elif text.isupper() and len(text) > 10:
        return "angry"
    
    # Keyword matching
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    
    return "neutral"

def main():
    st.title("🤖 Soul Sync Chat")
    st.markdown("*An advanced emotion-aware chatbot for meaningful conversations*")

    # Sidebar with better model selection
    st.sidebar.header("🎛️ Model Configuration")
    
    # Better model options
    model_options = {
        "DialoGPT Large (355M)": "microsoft/DialoGPT-large",
        "BlenderBot 3B": "facebook/blenderbot-3B",
        "BlenderBot 1B": "facebook/blenderbot-1B-distill",
        "GPT-2 Large (774M)": "gpt2-large",
        "GPT-2 XL (1.5B)": "gpt2-xl"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        list(model_options.keys()),
        index=0,
        help="Larger models provide better conversations but need more memory"
    )
    
    model_name = model_options[selected_model]
    
    # Load model with error handling
    try:
        with st.spinner(f"Loading {selected_model}..."):
            tokenizer, model, device = init_model(model_name)
        st.sidebar.success(f"✅ Model loaded on {device}")
        
        # Show model info
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            st.sidebar.info(f"GPU Memory: {memory_used:.1f} GB")
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        st.error("Please try a smaller model or check your system resources.")
        st.stop()

    # Enhanced generation settings
    st.sidebar.header("⚙️ Generation Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.6, 1.2, 0.8, 0.1, 
                               help="Higher = more creative, Lower = more focused")
        top_k = st.slider("Top-k", 30, 100, 50, 10,
                         help="Number of top tokens to consider")
    
    with col2:
        top_p = st.slider("Top-p", 0.7, 0.95, 0.9, 0.05,
                         help="Nucleus sampling threshold")
        max_tokens = st.number_input("Max tokens", 20, 100, 50, 10,
                                   help="Maximum response length")

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.quality_metrics = {}

    # Chat interface
    st.header("💬 Conversation")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, bot_msg, emotion) in enumerate(st.session_state.messages):
            # User message
            with st.chat_message("user"):
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "frustrated": "😤",
                    "confused": "🤔", "excited": "🤩", "anxious": "😰", "tired": "😴"
                }.get(emotion, "💬")
                st.write(f"{emotion_emoji} {user_msg}")
            
            # Bot response
            with st.chat_message("assistant"):
                st.write(bot_msg)

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Detect emotion
        emotion = detect_emotion_from_text(user_input)
        
        # Generate response
        with st.spinner("Thinking..."):
            # Build prompt with history
            history = st.session_state.conversation_history
            prompt = build_prompt(
                history=history,
                user_input=user_input,
                emotion_label=emotion,
                model_type=model_name,
                tokenizer=tokenizer
            )
            
            # Generate response
            response = generate_response(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                history=history,
                user_input=user_input,
                emotion=emotion
            )
            
            # Validate and fallback if needed
            is_valid, final_response = validate_response(response, user_input, emotion)
            if not is_valid:
                final_response = get_smart_fallback(user_input, emotion)
        
        # Update conversation history
        st.session_state.conversation_history.append((user_input, final_response))
        st.session_state.messages.append((user_input, final_response, emotion))
        
        # Truncate history if needed
        st.session_state.conversation_history = truncate_history(
            st.session_state.conversation_history, 
            tokenizer,
            max_tokens=800
        )
        
        # Update quality metrics
        st.session_state.quality_metrics = detect_conversation_quality(
            st.session_state.conversation_history
        )
        
        # Rerun to show new message
        st.rerun()

    # Sidebar stats
    if st.session_state.messages:
        st.sidebar.header("📊 Session Stats")
        
        # Basic stats
        num_exchanges = len(st.session_state.messages)
        emotions = [msg[2] for msg in st.session_state.messages]
        most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
        
        st.sidebar.metric("Messages", num_exchanges)
        st.sidebar.metric("Dominant Emotion", most_common_emotion.title())
        
        # Quality metrics
        if st.session_state.quality_metrics:
            quality = st.session_state.quality_metrics.get("quality", "unknown")
            st.sidebar.metric("Conversation Quality", quality.title())
            
            # Show suggestions
            suggestions = st.session_state.quality_metrics.get("suggestions", [])
            if suggestions:
                st.sidebar.write("**Suggestions:**")
                for suggestion in suggestions:
                    st.sidebar.write(f"• {suggestion}")

    # Control buttons
    st.sidebar.header("🔧 Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.quality_metrics = {}
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_text = "\n".join([
                    f"User: {msg[0]}\nBot: {msg[1]}\nEmotion: {msg[2]}\n---"
                    for msg in st.session_state.messages
                ])
                st.download_button(
                    "Download Chat",
                    chat_text,
                    "soul_sync_chat.txt",
                    "text/plain"
                )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Soul Sync v1.0**")
    st.sidebar.markdown("Emotion-aware conversational AI")

if __name__ == "__main__":
    main()