import streamlit as st
import torch
from llm_utils import load_model, generate_response, get_smart_fallback
from chat_utils import build_prompt, truncate_history, validate_response

st.set_page_config(page_title="Soul Sync Chat", layout="centered")

@st.cache_resource
def init_model(model_name: str):
    return load_model(model_name)

def main():
    st.title("🤖 Soul Sync Chat")
    st.markdown("*An improved emotion-aware chatbot for meaningful conversations*")

    # Sidebar controls with better defaults
    st.sidebar.header("Model Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",  # Added smaller option
            "distilgpt2", 
            "gpt2"
        ], 
        index=0
    )
    
    try:
        tokenizer, model, device = init_model(model_name)
        st.sidebar.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

    # Optimized generation parameters
    st.sidebar.header("Generation Settings")
    temperature = st.sidebar.slider("Temperature", 0.5, 1.0, 0.7, 0.1)
    top_p = st.sidebar.slider("Top-p", 0.3, 0.95, 0.8, 0.05)
    top_k = st.sidebar.slider("Top-k", 20, 80, 40, 10)
    max_new_tokens = st.sidebar.number_input(
        "Max new tokens", min_value=10, max_value=50, value=25, step=5
    )

    st.sidebar.header("Emotion Context")
    emotion_label = st.sidebar.selectbox(
        "Detected Emotion",
        ["neutral", "happy", "excited", "sad", "frustrated", "confused", "angry"], 
        index=0
    )

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.conversation_context = ""

    # Clear chat button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.session_state.conversation_context = ""
            st.rerun()
    
    with col2:
        debug_mode = st.checkbox("Debug Mode")

    # Display chat history
    st.subheader("💬 Conversation")
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.history:
            st.info("👋 Start a conversation! Try asking me something or just say hello.")
        
        for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
            # User message
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Bot message
            with st.chat_message("assistant"):
                st.write(bot_msg)

    # Input handling function
    def submit():
        user_input = st.session_state.user_input.strip()
        if not user_input:
            return

        # Show processing indicator
        with st.spinner("🤖 Soul Sync is thinking..."):
            # Truncate history for better performance
            safe_hist = truncate_history(
                st.session_state.history, tokenizer, max_tokens=200
            )
            
            # Build prompt with improved formatting
            prompt = build_prompt(
                safe_hist, 
                user_input,
                emotion_label=emotion_label,
                model_type="dialogpt" if "DialoGPT" in model_name else "gpt",
                tokenizer=tokenizer
            )
            
            # Debug information
            if debug_mode:
                with st.sidebar.expander("🔍 Debug Info", expanded=False):
                    st.write(f"**Prompt length:** {len(tokenizer.encode(prompt))} tokens")
                    st.text_area("Generated prompt:", prompt, height=100)
                    st.write(f"**History length:** {len(safe_hist)} exchanges")

            # Generate response
            bot_resp = generate_response(
                prompt,
                tokenizer,
                model,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                history=safe_hist,
                user_input=user_input
            )
            
            # Validate response and use fallback if needed
            is_valid, validated_response = validate_response(bot_resp, user_input)
            
            if not is_valid:
                # Use smart fallback for better responses
                bot_resp = get_smart_fallback(user_input, emotion_label)
                if debug_mode:
                    st.sidebar.warning("🔄 Used fallback response")
            else:
                bot_resp = validated_response
        
        # Update history
        st.session_state.history.append((user_input, bot_resp))
        st.session_state.user_input = ""
        st.rerun()

    # Input interface
    st.divider()
    
    # Text input
    st.text_input(
        "💭 Type your message:", 
        key="user_input", 
        on_change=submit,
        placeholder="Ask me anything or just say hello...",
        help="Press Enter to send your message"
    )
    
    # Quick action buttons
    st.subheader("🚀 Quick Start")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("👋 Say Hello"):
            st.session_state.user_input = "Hello! How are you?"
            submit()
    
    with col2:
        if st.button("❓ Ask Question"):
            st.session_state.user_input = "What is the capital of India?"
            submit()
    
    with col3:
        if st.button("😊 I'm Happy"):
            st.session_state.user_input = "I'm feeling really happy today!"
            submit()
    
    with col4:
        if st.button("💭 Random Chat"):
            import random
            random_prompts = [
                "Tell me something interesting",
                "How can you help me?",
                "What should we talk about?",
                "I'm looking for someone to chat with"
            ]
            st.session_state.user_input = random.choice(random_prompts)
            submit()

    # Tips section
    with st.expander("💡 Tips for Better Conversations"):
        st.markdown("""
        **For better responses:**
        - Be specific with your questions
        - Try asking about facts, feelings, or general topics
        - Use the emotion selector to get more contextual responses
        - Keep conversations focused on one topic at a time
        
        **If you get weird responses:**
        - Try rephrasing your question
        - Clear the chat history
        - Adjust the temperature (lower = more focused, higher = more creative)
        """)

    # Model info
    with st.expander("🔧 Model Information"):
        st.markdown(f"""
        **Current Model:** {model_name}
        **Device:** {device}
        **Parameters:** Temperature={temperature}, Top-p={top_p}, Top-k={top_k}
        **Max Tokens:** {max_new_tokens}
        
        **DialoGPT Models:** Better for conversational responses but may struggle with factual questions
        **GPT-2 Models:** Better for general knowledge but less conversational
        """)

if __name__ == "__main__":
    main()